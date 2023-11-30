from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import numpy as np
from alpha import run_snp_forecast
from matplotlib.ticker import MaxNLocator

import pandas as pd
import matplotlib.pyplot as plt
from beta import beta_model, best_stock


class RiskLevelApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=10)

        # Risk Level Slider
        label = Label(text='Risk Level:')
        self.layout.add_widget(label)

        self.risk_slider = Slider(min=0, max=100, value=50)
        self.layout.add_widget(self.risk_slider)


        # hundred_label = Label(text='100', size_hint=(None, None), size=(50, 50), pos=(10, 10))
        # self.layout.add_widget(hundred_label)
        #
        # five_hundred_label_right = Label(text='500', size_hint=(None, None), size=(50, 50))
        # self.layout.add_widget(five_hundred_label_right)  # Add label to the right of the slider
        #

        # Stock Symbols Input
        symbol_label = Label(text='Enter Stock Symbols (comma-separated):')
        self.layout.add_widget(symbol_label)

        self.symbol_input = TextInput(multiline=False)
        self.layout.add_widget(self.symbol_input)

        # Initial Capital Input
        capital_label = Label(text='Enter Initial Capital:')
        self.layout.add_widget(capital_label)

        self.capital_input = TextInput(multiline=False)
        self.layout.add_widget(self.capital_input)

        # Submit Button
        submit_button = Button(text='Submit')
        submit_button.bind(on_press=self.get_risk_symbols_and_capital)
        self.layout.add_widget(submit_button)

        # Result Label
        self.result_label = Label(text='', size_hint_y=None, height=100)
        self.layout.add_widget(self.result_label)


        # Result Label
        self.drawdown = Label(text='', size_hint_y=None, height=100)
        self.layout.add_widget(self.drawdown)


        # Result Label
        self.volume = Label(text='', size_hint_y=None, height=100)
        self.layout.add_widget(self.volume)


        # Result Label
        self.lowest = Label(text='', size_hint_y=None, height=100)
        self.layout.add_widget(self.lowest)


        return self.layout



    def get_risk_symbols_and_capital(self, instance):
        self.selected_risk = self.risk_slider.value
        self.stock_symbols = self.symbol_input.text.split(',')
        self.initial_capital = float(self.capital_input.text) if self.capital_input.text else 0.0
        print(f"Selected Risk Level: {self.selected_risk}")
        print(f"Stock Symbols: {self.stock_symbols}")
        print(f"Initial Capital: {self.initial_capital}")

        print('selected risk, ', self.selected_risk)


        # model_name = "LDA"
        model_name = "LDA_BAGG"
        # model_name = "HYBRID"

        trade_volume = 100 + (1000 - 100)/100 *  self.selected_risk * self.initial_capital / 10000



        # Get the optimal allocation ratios suggested by the beta model
        weights = beta_model(self.stock_symbols)


        equity_curve_list = list()
        stock_close_list = list()

        for i in range(len(self.stock_symbols)):
            # compute the trade volume and capital allocated to this symbol
            _trade_volume = trade_volume * weights[i]
            _initial_capital = self.initial_capital * weights[i]

            _symbol_list = []
            _symbol_list.append(self.stock_symbols[i])

            total_gain, returns, equity_curve, drawdown = run_snp_forecast(_symbol_list,
                                                                           _initial_capital, round(_trade_volume),
                                                                           model_name)

            returns_numpy = returns.to_numpy()
            equity_curve_numpy = equity_curve.to_numpy()

            equity_curve_zero_mean = (equity_curve_numpy - 1)[1:]


        #
        # total_gain, returns, equity_curve, drawdown = run_snp_forecast(self.stock_symbols,
        #                                                                self.initial_capital, round(trade_volume), model_name)
        #
        # returns_numpy = returns.to_numpy()
        # equity_curve_numpy = equity_curve.to_numpy()
        #
        # equity_curve_zero_mean = (equity_curve_numpy - 1)[1:]




            # Update the result_label with the total_gain
            self.result_label.text = f"Total Gain: {total_gain[1]}"
            self.drawdown.text = f"Drawdown: {round(drawdown * 100, 2)}%"
            self.volume.text=  f"Trade Volume: {round(trade_volume)}"
            self.lowest.text=  f"Lowest Equity: {round(np.min(equity_curve_zero_mean * 100))}%"



            df = pd.read_csv(f'data/{self.stock_symbols[i]}.csv')
            stock_close = df['adj_close'].to_numpy()


            stock_close_norm= (stock_close - stock_close[0])/stock_close[0]

            stock_close_rat = stock_close/stock_close[0]

            equity_curve_list.append(equity_curve_numpy)
            stock_close_list.append(stock_close_rat)







        # fig, ax = plt.subplots(figsize=(6, 4))  # Fix: Use plt.subplots instead of plt.figure
        # ax.plot((equity_curve_numpy - 1) * 100, label="Algo")  # Add legend labels
        # ax.plot((stock_close_norm ) * 100, label=f"{self.stock_symbols[0]}")  # Plot SPY data
        #
        #
        #
        # plt.xlabel('Time (Days)')
        # plt.ylabel('Equity Curves (%)')
        # plt.title(f'Equity Curve Over Time: Algo vs. {self.stock_symbols[0]}')
        # plt.legend()  # Include legend
        # plt.show()
        #
        # # plt.xticks(xticks)
        # # plt.xticklabels(xticklabels)
        #
        # # # Ensure that only integer ticks are displayed on the y-axis
        # # plt.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.savefig('chart.png')  # Save the chart as an image file
        # # # Display the Matplotlib chart
        # plt.show(block=True)

        sum_equity = 0
        sum_stock = 0

        for i in range(len(self.stock_symbols)):
            _initial_capital = self.initial_capital * weights[i]
            equity_curve = equity_curve_list[i]
            stock_close_rat = stock_close_list[i]

            sum_equity += equity_curve * _initial_capital
            sum_stock += stock_close_rat * _initial_capital

        #
        # best_stoc = best_stock(self.stock_symbols)
        # best_stock_close = df['adj_close'].to_numpy()
        # df_ = pd.read_csv(f'data/{self.stock_symbols[i]}.csv')
        # stock_close = df['adj_close'].to_numpy()
        #
        # stock_close_norm = (stock_close - stock_close[0]) / stock_close[0]
        #
        # stock_close_rat = stock_close / stock_close[0]
        #
        # total_gain_best, returns_best, equity_curve_best, drawdown_best = run_snp_forecast([best_stoc],
        #                                                                self.initial_capital, round(trade_volume),
        #                                                                model_name)


        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sum_equity, label=f"Algo - Combined Stocks - Alpha: {model_name}")
        ax.plot(sum_stock, label=f"Combined Stocks")  # Plot combined stocks

        # ax.plot(equity_curve_best.to_numpy() * self.initial_capital, label=f"Algo- Best Stock")  # Plot combined stocks

        plt.xlabel('Time (Days)')
        plt.ylabel('Equity Curves ($)')
        plt.title(f'Equity Curve Over Time: Algo vs. {self.stock_symbols[0]}')
        plt.legend()  # Include legend
        plt.show()

        # plt.xticks(xticks)
        # plt.xticklabels(xticklabels)
        plt.savefig('chart.png')  # Save the chart as an image file
        plt.show(block=True)


app = RiskLevelApp()
app.run()