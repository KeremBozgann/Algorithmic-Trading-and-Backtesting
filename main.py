from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import matplotlib.pyplot as plt
import numpy as np
from alpha import run_snp_forecast
from matplotlib.ticker import MaxNLocator



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

        trade_volume = 100 + (1000 - 100)/100 *  self.selected_risk
        total_gain, returns, equity_curve, drawdown = run_snp_forecast(self.stock_symbols, self.initial_capital, round(trade_volume))
        returns_numpy = returns.to_numpy()
        equity_curve_numpy = equity_curve.to_numpy()

        equity_curve_zero_mean = (equity_curve_numpy - 1)[1:]


        # Update the result_label with the total_gain
        self.result_label.text = f"Total Gain: {total_gain[1]}"
        self.drawdown.text = f"Drawdown: {round(drawdown * 100, 2)}%"
        self.volume.text=  f"Trade Volume: {round(trade_volume)}"
        self.lowest.text=  f"Lowest Equity: {round(np.min(equity_curve_zero_mean * 100))}%"

        import pandas as pd
        import matplotlib.pyplot as plt

        # Assuming 'equity_curve_numpy' is already defined
        # Assuming 'spy_close' is defined and loaded from the CSV file

        df = pd.read_csv('data/SPY.csv')
        spy_close = df['adj_close'].to_numpy()

        spy_close_norm= (spy_close - spy_close[0])/spy_close[0]

        fig, ax = plt.subplots(figsize=(6, 4))  # Fix: Use plt.subplots instead of plt.figure
        ax.plot((equity_curve_numpy - 1) * 100, label="Algo")  # Add legend labels
        ax.plot((spy_close_norm ) * 100, label="AAPL")  # Plot SPY data



        plt.xlabel('Time (Days)')
        plt.ylabel('Equity Curves (%)')
        plt.title('Equity Curve Over Time: Algo vs. SPY')
        plt.legend()  # Include legend
        plt.show()

        # plt.xticks(xticks)
        # plt.xticklabels(xticklabels)

        # # Ensure that only integer ticks are displayed on the y-axis
        # plt.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('chart.png')  # Save the chart as an image file
        # # Display the Matplotlib chart
        plt.show(block=True)


app = RiskLevelApp()
app.run()