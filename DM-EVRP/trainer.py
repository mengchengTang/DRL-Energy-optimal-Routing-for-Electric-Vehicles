import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import sys
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.reinforce_baselines import ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline, StateCritic
from utils import torch_load_cpu, move_to, plot_delivery_graph
import math
import xlwt
import csv
from datetime import timedelta
from utils.data_utils import save_dataset
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# ------------------------------------------valid----------------------------------------------------------------------
def validate(data_loader, actor, render_fn, num_nodes, charging_num, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    actor.set_decode_type("greedy")

    rewards = []
    for batch_idx, batch in enumerate(data_loader):
        x = batch
        static, dynamic, = batch[0:2]
        static = static.float().to(device)
        dynamic = dynamic.float().to(device)

        with torch.no_grad():
            tour_indices, _, R= actor.forward(x)
        reward = R.sum(1).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png' % (batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path, dynamic, num_nodes, charging_num, batch_idx)

    actor.train()
    return np.mean(rewards)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
     Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train(actor, baseline, optimizer, lr_scheduler, task, num_nodes, train_data_out, valid_data,
          render_fn, batch_size, max_grad_norm,iterations,baselines,charging_num,
          **kwargs):
    """Constructs the main actor  networks, and performs all training."""
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join("ExperimentalLog", "train", f"{num_nodes}", f"{baselines}", f"C{num_nodes}_{now}")

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf

    epoch_reward = []
    epoch_loss = []

    out_path = os.path.join("ExperimentalData", "train_data", f"{num_nodes}", f"{baselines}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_path_epoch = os.path.join(out_path, f"Epoch_C{num_nodes}_{now}.csv")
    out_path_batch = os.path.join(out_path, f"Batch_C{num_nodes}_{now}.csv")

    for epoch in range(iterations):  # train epoch
        train_data = baseline.wrap_dataset(train_data_out)
        train_loader = DataLoader(train_data, batch_size, True, num_workers=0)

        actor.train()
        actor.set_decode_type("sample")

        times, losses, rewards, critic_rewards= [], [], [], []
        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):  # train batch

            x, bl_val = baseline.unwrap_batch(batch)
            bl_val = move_to(bl_val, device) if bl_val is not None else None
            tour_indices, tour_logp, R= actor(x)
            reward = torch.sum(R, dim=1)  # R[batch,sequence_len] -> reward[batch]

            bl_val, bl_loss = baseline.eval(x, reward) if bl_val is None else (bl_val, 0)

            advantage = (reward - bl_val)
            reinforce_loss = (advantage.detach() * tour_logp.sum(dim=1)).mean()
            loss = reinforce_loss + bl_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norms = clip_grad_norms(optimizer.param_groups, max_grad_norm)
            optimizer.step()

            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(reinforce_loss.detach()).item())


            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                with open(out_path_batch, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([mean_reward, mean_loss])

                print('Batch %d/%d, reward: %2.3f,  loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward,  mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        mean_time =np.mean(times)

        epoch_reward.append(mean_reward)
        epoch_loss.append(mean_loss)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        torch.save(
            {
                'model': actor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(epoch_dir, 'epoch-{}.pt'.format(epoch))
        )


        valid_dir = os.path.join(save_dir, '%s' % epoch)
        mean_valid = validate(valid_loader, actor,  render_fn, num_nodes, charging_num,
                              valid_dir, num_plot=1)


        with open(out_path_epoch, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([mean_reward, mean_loss, mean_time, mean_valid])

        baseline.epoch_callback(actor, epoch)
        # lr_scheduler should be called at end of epoch
        lr_scheduler.step()


        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'best.pt')
            print('Saving best model and state...')
            torch.save(
                {
                    'model': actor.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    'baseline': baseline.state_dict()
                },
                save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
               mean_time))

    return epoch_reward, epoch_loss


def train_EVRP(args):
    """
    train
    """
    from problems.EVRP import VehicleRoutingDataset
    if args.model == "DRL":
        from nets.DRLModel import AttentionModel
    elif args.model == "AM":
        from nets.AM import AttentionModel
    elif args.model == "pointer":
        from nets.PointNetwork import DRL4EVRP
    else:
        print("Please enter a correct network name")
        sys.exit("Error message, program terminated.")

    # Determines the maximum amount of load for a vehicle based on num nodes
    MAX_DEMAND = 4
    STATIC_SIZE = 2    # (x, y)
    DYNAMIC_SIZE = 4   # (load, demand,soc,time)
    max_load = 4       # BYD vans 4000KG

    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       args.t_limit,
                                       args.Start_SOC,
                                       args.velocity,
                                       max_load,
                                       MAX_DEMAND,
                                       args.charging_num,
                                       args.seed,
                                       args)

    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       args.t_limit,
                                       args.Start_SOC,
                                       args.velocity,
                                       max_load,
                                       MAX_DEMAND,
                                       args.charging_num,
                                       args.seed + 2,
                                       args)

    if args.model == "pointer":
        actor = DRL4EVRP(STATIC_SIZE,
                         DYNAMIC_SIZE,
                         args.hidden_size,
                         train_data.update_dynamic,
                         train_data.update_mask,
                         args.num_layers,
                         args.dropout).to(device)
    elif args.model == "attention":
        actor = AttentionModel(
            args.embedding_dim,
            args.hidden_size,
            args,
            n_encode_layers=args.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=args.normalization,
            tanh_clipping=args.tanh_clipping,
            update_dynamic=train_data.update_dynamic,
            update_mask=train_data.update_mask,
        ).to(device)
    else:
        raise ValueError('choose the right model ')

    load_data = {}
    if args.checkpoint:
        load_path = os.path.join(args.checkpoint)
        if load_path is not None:
            print('  [*] Loading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)

    actor.load_state_dict({**actor.state_dict(), **load_data.get('model', {})})

    if not args.test:  # train
        if args.baselines == 'exponential':
            baseline = ExponentialBaseline(args.exp_beta)
        elif args.baselines == 'critic':
            baseline = CriticBaseline(
                StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
            )
        elif args.baselines == 'rollout':
            baseline = RolloutBaseline(actor, valid_data, args)

        if args.bl_warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, args.bl_warmup_epochs, warmup_exp_beta=args.exp_beta)

        # Load baseline from data, make sure script is called with same type of baseline
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])
        # Initialize the optimizer
        optimizer = optim.Adam(
            [{'params': actor.parameters(), 'lr': args.actor_lr}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': args.critic_lr}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: args.lr_decay ** epoch)

        kwargs = vars(args)
        kwargs['optimizer'] = optimizer
        kwargs['lr_scheduler'] = lr_scheduler
        kwargs['train_data_out'] = train_data
        kwargs['valid_data'] = valid_data
        kwargs['render_fn'] = plot_delivery_graph
        # train
        epoch_reward, epoch_loss= train(actor, baseline, **kwargs)
        # save data to excel
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet(f'train_{args.baselines}_C{args.num_nodes}', cell_overwrite_ok=True)
        col = ('Average reward per epoch', 'Average loss per epoch')
        end = '%s' % datetime.datetime.now().time()
        end = end.replace(':', '_')
        for i in range(0, 2):
            sheet.write(0, i, col[i])
        for i in range(0, len(epoch_reward)):
            sheet.write(i + 1, 0, epoch_reward[i])
            sheet.write(i + 1, 1, epoch_loss[i])
        save_path = os.path.join("ExperimentalData", 'train_data', f"{args.num_nodes}", f"{args.baselines}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, f'trainC{args.num_nodes}_{end}.xls')
        book.save(save_path)

    else:  # test
        if args.test_file:
            print("Load test data")
            test_data = EVRPDataset(args.test_file, num_samples=1, offset=0)
        else:
            print("generate test data")
            test_data = VehicleRoutingDataset(args.test_size,
                                              args.num_nodes,
                                              args.t_limit,
                                              args.Start_SOC,
                                              args.velocity,
                                              max_load,
                                              MAX_DEMAND,
                                              args.charging_num,
                                              args.test_seed,
                                              args)
            # Save data to file for testing and comparison
            test_dataloader = DataLoader(test_data, args.test_size, False, num_workers=0)
            for i in test_dataloader:
                data = i
            static, dynamic, distances, slope = data
            thedata = list(zip(static.tolist(),           # [2, sequence_len] coordinate information
                               dynamic.tolist(),          # [4, sequence_len] Dynamic information: loads, demands, SOC, time
                               distances.tolist(),        # [sequence_len, sequence_len]
                               slope.tolist()             # [sequence_len, sequence_len]
                               ))
            if args.CVRP_lib_test:
                filepath = os.path.join("ExperimentalData", "CVRPlib")
                name = args.CVRP_lib_path.split("/")[-1]
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                filename = os.path.join(filepath, name)
            else:
                filepath = os.path.join("ExperimentalData", "test_data", f"{args.num_nodes}")
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                filename = os.path.join(filepath, f'{args.test_size}_seed{args.test_seed}.pkl')
            save_dataset(thedata, filename)
        widths = args.width if args.width is not None else [0]  # list
        for width in widths:
            mean_costs, duration = eval_dataset(test_data, width, args.softmax_temperature, args, actor, plot_delivery_graph)


def eval_dataset(test_date, width, softmax_temp, args, actor, render):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model = actor
    dataset = test_date
    results = _eval_dataset(model, dataset, width, softmax_temp, args, render)

    parallelism = args.eval_batch_size
    costs, durations, energy = zip(*results)
    costs = torch.cat(costs, dim=0)
    energys = torch.cat(energy, dim=0)
    costs = costs.cpu().numpy()
    energys = energys.cpu().numpy()
    # save to file
    if not args.CVRP_lib_test:
        now_time = '%s' % datetime.datetime.now().time()
        now_time = now_time.replace(':', '_')
        output_path = os.path.join("ExperimentalLog", "test", f"{args.num_nodes}", "data_record")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, f"{now_time}.csv")
        with open(output_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(costs)):
                writer.writerow([costs[i], durations[i], energys[i]])
            writer.writerow("####### mean value ###########")
            writer.writerow([np.mean(costs), np.mean(durations), np.mean(energys)])

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average energy: {} +- {}".format(np.mean(energys), 2 * np.std(energys) / np.sqrt(len(energys))))
    print("Average batch duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))
    return np.mean(costs), durations


def _eval_dataset(model, dataset, width, softmax_temp, args, render=None):

    model.eval()
    model.set_decode_type(
        "greedy" if args.decode_strategy in ('bs', 'greedy') else "sample")
    dataloader = DataLoader(dataset, args.eval_batch_size, False, num_workers=0)

    results = []
    for batch_idx, batch in enumerate(dataloader):
        start = time.time()
        with torch.no_grad():
            if args.decode_strategy in ('sample', 'greedy'):
                if args.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert args.eval_batch_size <= args.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * args.eval_batch_size > args.max_calc_batch_size:
                    assert args.eval_batch_size == 1
                    assert width % args.max_calc_batch_size == 0
                    batch_rep = args.max_calc_batch_size
                    iter_rep = width // args.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                costs, min_sequence, energy = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
            else:
                assert args.decode_strategy == 'bs'
                costs, min_sequence, energy = model.beam_search(batch, beam_width=width)
        duration = time.time() - start
        results.append((costs, duration, energy))
        if render is not None and batch_idx < args.plot_num:
            static, dynamic, _, _ = batch
            name = f'batch%d_%2.4f.png' % (batch_idx, costs[0].item())
            if not args.CVRP_lib_test:
                path = os.path.join("ExperimentalLog", "test", f"{args.num_nodes}", "graph")
            else:
                path = os.path.join("ExperimentalLog", "test", "CVRPlib")
            if not os.path.exists(path):
                os.makedirs(path)
            save_path =os.path.join(path, name)
            render(static.cpu(), min_sequence.cpu(), save_path, dynamic.cpu(), args.num_nodes, args.charging_num, batch_idx)
    return results


def make_instance(i):
    static, dynamic, distance, slope = i
    return (torch.tensor(static).to(device),
            torch.tensor(dynamic).to(device),
            torch.tensor(distance).to(device),
            torch.tensor(slope).to(device))


def EVRPDataset(filename=None,  num_samples=256, offset=0):
    assert os.path.splitext(filename)[1] == '.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    data = [make_instance(i) for i in data[offset:offset + num_samples]]
    return data

