from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
import fitz  # Import pyMuPDF

class PDFNoteTakerApp(App):
    def build(self):
        # Create a BoxLayout to hold the PDF viewer, text input, and file chooser
        layout = BoxLayout(orientation='horizontal', spacing=10)

        # Create the PDF viewer widget
        pdf_viewer = ScrollView(size_hint=(0.3, 1))
        pdf_widget = Label(text="No PDF selected")

        # Create a text input widget for taking notes
        note_input = TextInput(hint_text="Take notes here", size_hint=(0.4, 1))

        # Create a FileChooser widget to select PDF files
        file_chooser = FileChooserListView(size_hint=(0.3, 1))
        file_chooser.bind(selection=self.load_pdf)

        # Add the PDF viewer, text input, and file chooser to the layout
        pdf_viewer.add_widget(pdf_widget)
        layout.add_widget(pdf_viewer)
        layout.add_widget(note_input)
        layout.add_widget(file_chooser)

        return layout

    def load_pdf(self, instance, value):
        if value:
            pdf_path = value[0]
            doc = fitz.open(pdf_path)
            pdf_text = ''
            for page_num in range(doc.page_count):
                page = doc[page_num]
                pdf_text += page.get_text()

            # Display the PDF content in the PDF viewer
            pdf_widget = self.root.children[0].children[0]
            pdf_widget.text = pdf_text

if __name__ == '__main__':
    PDFNoteTakerApp().run()
