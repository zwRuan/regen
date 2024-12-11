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

LOG_NAME="alpacafarm"
#LOG_NAME="analyze/encoder/mt5-large"
LOG_PATH="$LOG_PATH/$LOG_NAME/$TIME_STAMP"

mkdir -p $LOG_PATH
ts --set_logdir $LOG_PATH

ts -G 1 -L entropy -O entropy.log bash scripts/eval/alpaca_farm_entropy.sh
#ts -G 1 -L alpha -O alpha.log bash scripts/eval/alpaca_farm_alpha.sh