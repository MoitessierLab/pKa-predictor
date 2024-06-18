from ray import tune
from ray.air import Checkpoint, session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def load_data(file_name):
    with open(file_name, "rb") as f:
        conts = f.read()
    data = pickle.loads(conts)
    return data


def tune_hypers(config, checkpoint_dir=None):
    # Loading the dataset
    # print("Loading dataset...")
    train_dataset = load_data('/content/train.pickle')
    test_dataset = load_data('/content/test.pickle')

    # Prepare training
    train_loader = DataLoader(train_dataset, config["batch_size"],
                              num_workers=0, shuffle=True)
    test_loader = DataLoader(test_dataset, config["batch_size"],
                             num_workers=0, shuffle=True)

    # Loading the model
    # print("Loading model...")
    model_params = {k: v for k, v in config.items() if k.startswith("model_")}
    model = GNN(feature_size=train_dataset[0].x.shape[1],
                edge_dim=train_dataset[0].edge_attr.shape[1],
                model_params=model_params)
    model = model.to(device)
    # print(f"Number of parameters: {count_parameters(model)}")

    # < 1 increases precision, > 1 recall
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["scheduler_gamma"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Start training
    for epoch in range(500):
        # Training
        model.train()
        train_loss = train(epoch, model, train_loader, optimizer, loss_fn)
        # print(f"Epoch {epoch} | Train Loss {loss}", end='')

        # Validation
        model.eval()
        val_loss = evaluate(epoch, model, test_loader, loss_fn)
        # print(f"Epoch {epoch} | Test Loss {loss}")

        scheduler.step()

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(train_loss=(train_loss), val_loss=(val_loss))