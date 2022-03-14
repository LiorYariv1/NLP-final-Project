from anvil import *
from ._anvil_designer import Form1Template

class Form1(Form1Template):

  def _init_(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)

    # Any code you write here will run when the form opens.

  def check_box_1_copy_change(self, **event_args):
    """This method is called when this checkbox is checked or unchecked"""
    pass