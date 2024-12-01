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

LOG_NAME="entropy/FIRST_N_TOKENS"
#LOG_NAME="analyze/encoder/mt5-large"
LOG_PATH="$LOG_PATH/$LOG_NAME/$TIME_STAMP"

mkdir -p $LOG_PATH
ts --set_logdir $LOG_PATH
FIRST_N_TOKENS=(1000 900 800 700)
ts -G 1 -L entropy -O entropy_first_n_tokens.log bash scripts/eval/alpaca_farm.sh $FIRST_N_TOKENS 1
FIRST_N_TOKENS=(600 500 450 400 350)
ts -G 1 -L entropy -O entropy_first_n_tokens.log bash scripts/eval/alpaca_farm.sh $FIRST_N_TOKENS 2
FIRST_N_TOKENS=(300 250 200 150 100 50)
ts -G 1 -L entropy -O entropy_first_n_tokens.log bash scripts/eval/alpaca_farm.sh $FIRST_N_TOKENS 3