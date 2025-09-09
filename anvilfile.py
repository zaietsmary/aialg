from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run before the form opens.

  def file_loader_1_change(self, file, **event_args):
    if not file:
      return

    self.image_1.source = file
    file_name = getattr(file, "name", "файл")
    self.text_area_1.text = f"Файл {file_name} завантажено!\nРозмір: {len(file.get_bytes())} байт"

    try:
      vector, norm_vector, bw_file = anvil.server.call('process_image', file)
      self.image_1.source = bw_file
      self.text_area_vector.text = str(vector)
      self.text_area_norm.text = str(norm_vector)
    except Exception as e:
      self.text_area_1.text = f"Помилка при обробці зображення: {e}"




