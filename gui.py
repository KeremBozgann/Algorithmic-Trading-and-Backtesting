from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

class RiskLevelApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=10)

        # Risk Level Slider
        label = Label(text='Risk Level:')
        self.layout.add_widget(label)

        self.risk_slider = Slider(min=0, max=100, value=50)
        self.layout.add_widget(self.risk_slider)

        # Stock Symbols Input
        symbol_label = Label(text='Enter Stock Symbols (comma-separated):')
        self.layout.add_widget(symbol_label)

        self.symbol_input = TextInput(multiline=False)
        self.layout.add_widget(self.symbol_input)

        # Submit Button
        submit_button = Button(text='Submit')
        submit_button.bind(on_press=self.get_risk_and_symbols)
        self.layout.add_widget(submit_button)

        return self.layout

    def get_risk_and_symbols(self, instance):
        self.selected_risk = self.risk_slider.value
        self.stock_symbols = self.symbol_input.text.split(',')
        print(f"Selected Risk Level: {self.selected_risk}")
        print(f"Stock Symbols: {self.stock_symbols}")


app = RiskLevelApp()
app.run()

# Access selected_risk and stock_symbols here
print("Selected Risk Level in Main: ", app.selected_risk)
print("Stock Symbols in Main: ", app.stock_symbols)
