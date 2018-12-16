import argparse


class Configuration(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_frac", default=0.2, type=float, help="fractions of validation")
    parser.add_argument("--kpi_id", type=str, help="string id for kpi in csv files", default="KPI ID")
    parser.add_argument("--timestamp", type=str, default='timestamp')
    parser.add_argument("--value", type=str, default="value")
    parser.add_argument("--label", type=str, default="label")
    parser.add_argument("--mode", type=str, default="unsupervised")
    parser.add_argument("--have_label", type=int, default=1)

    parser.add_argument("--optim", type=str, default='sgd')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--ano_thresh", type=float, default=7.0)
    ### dataset setting #####
    parser.add_argument("--window_size", type=int, default=120)

    #### network setting #######
    parser.add_argument("--bidirection", type=int, default=0)
    #parser.add_argument("--model", type=str, default=)

    args = parser.parse_args()

    val_frac = args.val_frac
    ### loading file setting
    kpi_id = args.kpi_id
    timestamp = args.timestamp
    value = args.value
    label = args.label

    optim = args.optim
    lr = args.lr
    momentum = args.momentum
    batch_size = args.batch_size
    epochs = args.epochs

    window_size = args.window_size

    bidirection = bool(args.bidirection)
    hidden_size = [64, 128]
    seed = 0
    mode = args.mode
    have_label = bool(args.have_label)
    dropout_rate = args.dropout_rate
    ano_thresh = args.ano_thresh