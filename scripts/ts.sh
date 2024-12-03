# LOG_PATH="Logs"
# TIME_STAMP=$(date "+%Y-%m-%d_%H-%M-%S")

# LOG_NAME="entropy/threshold"
# #LOG_NAME="analyze/encoder/mt5-large"
# LOG_PATH="$OG_PATH/$LOG_NAME/$TIME_STAMP"

# mkdir -p $LOG_PATH
# ts --set_logdir $LOG_PATH

# ts -G 1 -L entropy -O entropy_threshold.log bash scripts/eval/alpaca_farm.sh

LOG_PATH="Logs"
TIME_STAMP=$(date "+%Y-%m-%d_%H-%M-%S")

LOG_NAME="entropy/METHOD"
#LOG_NAME="analyze/encoder/mt5-large"
LOG_PATH="$LOG_PATH/$LOG_NAME/$TIME_STAMP"

mkdir -p $LOG_PATH
ts --set_logdir $LOG_PATH

# ts -G 1 -L entropy -O all_log_softmax.log bash scripts/eval/alpaca_farm.sh all_log_softmax 1
# ts -G 1 -L entropy -O pos_neg_log_softmax.log bash scripts/eval/alpaca_farm.sh pos_neg_log_softmax 2
ts -G 1 -L entropy -O all_softmax.log bash scripts/eval/alpaca_farm.sh all_softmax 3
# ts -G 1 -L entropy -O pos_neg_softmax.log bash scripts/eval/alpaca_farm.sh pos_neg_softmax 4
# ts -G 1 -L entropy -O base_logits.log bash scripts/eval/alpaca_farm.sh base_logits 5

# ts -G 1 -L entropy -O all_log_softmax_entropy.log bash scripts/eval/alpaca_farm_entropy.sh all_log_softmax
# ts -G 1 -L entropy -O pos_neg_log_softmax_entropy.log bash scripts/eval/alpaca_farm_entropy.sh pos_neg_log_softmax
# ts -G 1 -L entropy -O all_softmax_entropy.log bash scripts/eval/alpaca_farm_entropy.sh all_softmax
# ts -G 1 -L entropy -O pos_neg_softmax_entropy.log bash scripts/eval/alpaca_farm_entropy.sh pos_neg_softmax
# ts -G 1 -L entropy -O base_logits_entropy.log bash scripts/eval/alpaca_farm_entropy.sh base_logits