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

LOG_NAME="alpacafarm/entropy/more_threshold_to_1"
#LOG_NAME="analyze/encoder/mt5-large"
LOG_PATH="$LOG_PATH/$LOG_NAME/$TIME_STAMP"

mkdir -p $LOG_PATH
ts --set_logdir $LOG_PATH

ts -G 1 -L entropy -O more_threshold_to_1.log bash scripts/eval/alpaca_farm_entropy.sh


LOG_PATH="Logs"
TIME_STAMP=$(date "+%Y-%m-%d_%H-%M-%S")

LOG_NAME="alpacafarm/entropy/more_threshold_to_2"
#LOG_NAME="analyze/encoder/mt5-large"
LOG_PATH="$LOG_PATH/$LOG_NAME/$TIME_STAMP"

mkdir -p $LOG_PATH
ts --set_logdir $LOG_PATH

ts -G 1 -L entropy -O more_threshold_to_2.log bash scripts/eval/alpaca_farm_entropy2.sh
