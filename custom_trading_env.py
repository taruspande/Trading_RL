import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Sell = 0
    Buy = 1
    Nothing = 2


class Positions(Enum):
    Short = 0
    Long = 1
    Hold = 2
    

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

# -------------------------------------------------------------------------------- #
    def __init__(self, df, window_size,frame_bound):
        assert df.ndim == 2
        self.frame_bound = frame_bound  
        
        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        self.risk_array = np.zeros(len(self.prices))
        self.rewards_array = np.zeros(len(self.prices)) 

        
        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self.Prev_position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

# -------------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
# -------------------------------------------------------------------------------- #
    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self.Prev_position = None
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()
        self.risk_array.fill(0.0)

# -------------------------------------------------------------------------------- #

    def step(self, action):
        self._done = False
        self._current_tick += 1
        
        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self.rewards_array[self._current_tick] = step_reward
        self._total_reward += step_reward
        risk_val = self.calculate_risk()
        self.risk_array[self._current_tick]=risk_val
        self._update_profit(action)

        trade = False
        if action!=Actions.Nothing.value:
            if ((action == Actions.Buy.value and (self._position == Positions.Short or (self.Prev_position == Positions.Short and self._position==Positions.Hold))) or(action == Actions.Sell.value and (self._position == Positions.Long or (self.Prev_position == Positions.Long and self._position==Positions.Hold) ))):
                trade = True

            if trade:
                if self._position!=Positions.Hold:
                    self.Prev_position  = self._position
                    
                if self._position!=Positions.Hold:
                    self._position = self._position.opposite()
                elif self._position==Positions.Hold and action==1:
                    self._position=Positions.Long
                else:
                    self._position=Positions.Short
                
                self._last_trade_tick = self._current_tick
                
            elif self._position==Positions.Hold :
                self._position = self.Prev_position
        else :
            if self._position!=Positions.Hold: self.Prev_position  = self._position
            self._position = Positions.Hold
            

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

# -------------------------------------------------------------------------------- #
    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

# -------------------------------------------------------------------------------- #
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)
# -------------------------------------------------------------------------------- #
    def calculate_risk(self):
        # Calculate the daily returns
        daily_returns = np.diff(self.prices) / self.prices[:-1]

        # Calculate the Sharpe Ratio
        # Assuming a risk-free rate of 0%
        risk_free_rate = 0.0

        # Calculate the excess daily returns
        excess_daily_returns = daily_returns - risk_free_rate

        # Calculate the average of the excess daily returns
        avg_excess_return = np.mean(excess_daily_returns)

        # Calculate the standard deviation of the excess daily returns
        std_dev_excess_return = np.std(excess_daily_returns)

        # Calculate the daily Sharpe Ratio
        daily_sharpe_ratio = avg_excess_return / std_dev_excess_return

        # Annualize the Sharpe Ratio
        annual_factor = np.sqrt(252)  # assuming 252 trading days in a year
        annual_sharpe_ratio = daily_sharpe_ratio * annual_factor

        # The Sharpe Ratio serves as a risk metric (higher is better)
        return -annual_sharpe_ratio  # Return as negative because gym environments generally expect to minimize the value


# -------------------------------------------------------------------------------- #

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            else :
                color = 'blue'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)
# -------------------------------------------------------------------------------- #

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.figure(figsize=(12, 6))  # Make the figure larger
        
        # Plot 1: Price and trades
        plt.subplot(2, 1, 1)
        plt.plot(self.prices)
        short_ticks = [tick for tick, pos in enumerate(self._position_history) if pos == Positions.Short]
        long_ticks = [tick for tick, pos in enumerate(self._position_history) if pos == Positions.Long]
        hold_ticks = [tick for tick, pos in enumerate(self._position_history) if pos == Positions.Hold]
        
        plt.plot(short_ticks, self.prices[short_ticks], 'ro', label='Short')
        plt.plot(long_ticks, self.prices[long_ticks], 'go', label='Long')
        plt.plot(hold_ticks, self.prices[hold_ticks], 'bo', label='Hold')  # Added 'Hold' positions in blue
        
        plt.legend()
        plt.title("Total Reward: %.6f" % self._total_reward + ' ~ ' + "Total Profit: %.6f" % self._total_profit)
        
        # Plot 2: Risk Over Time
        plt.subplot(2, 1, 2)
        plt.plot(self.risk_array)  # Replace with your risk array
        plt.title("Risk Over Time")
        plt.xlabel("Time")
        plt.ylabel("Risk")
        
        plt.tight_layout()  # Add this to ensure the plots are spaced nicely
        plt.show()



# -------------------------------------------------------------------------------- #
    def _process_data(self):
        prices = self.df.loc[:, 'open'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        prev_high = self.df.loc[:, 'prev_high'].to_numpy()
        prev_high = prev_high[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        SMA_5 = self.df.loc[:, 'SMA_5'].to_numpy()
        SMA_5 = SMA_5[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        EMA_5 = self.df.loc[:, 'EMA_5'].to_numpy()
        EMA_5 = EMA_5[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff,prev_high,SMA_5,EMA_5))

        return prices, signal_features
# -------------------------------------------------------------------------------- #
    
    def _calculate_reward(self, action):
        step_reward = 0
        trade = False

        penalty_for_extending_short = 50  # Feel free to adjust this value

        if ((action == Actions.Buy.value and (self._position == Positions.Short or (self.Prev_position == Positions.Short and self._position==Positions.Hold))) or(action == Actions.Sell.value and (self._position == Positions.Long or (self.Prev_position == Positions.Long and self._position==Positions.Hold)))):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long or (self.Prev_position==Positions.Long and self._position==Positions.Hold):
                step_reward += price_diff
            elif self._position == Positions.Short or (self.Prev_position==Positions.Short and self._position==Positions.Hold):
                step_reward -= penalty_for_extending_short  # Apply penalty for extending a short position

        if action==Actions.Nothing.value:
            step_reward = -1
        return step_reward

# -------------------------------------------------------------------------------- #
    
    
    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and (self._position == Positions.Short or (self.Prev_position == Positions.Short and self._position==Positions.Hold))) or(action == Actions.Sell.value and (self._position == Positions.Long or (self.Prev_position == Positions.Long and self._position==Positions.Hold) ))):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long or (self.Prev_position==Positions.Long and self._position==Positions.Hold):
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

# -------------------------------------------------------------------------------- #

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit    
