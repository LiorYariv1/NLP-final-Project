# A simple script to calculate BMI
from pywebio.input import input, TEXT,input_group, checkbox
from pywebio.output import put_text
from pywebio import pin
from pywebio.pin import pin as pin_obj
import pywebio
from T5 import PlotGenerationModel

class Pyweb():
    def __init__(self):
        self.p3_model = PlotGenerationModel('/home/student/project/model1902__kw_Rake_p3', 't5-base')
        self.submit = False

    def submission(self):
        self.submit=True

    def ui(self):
        pywebio.output.put_markdown(r""" <h1>Let's Create A Movie.</h1>""")
        pin.put_input('title', label='Choose your Movie Title')
        pin.put_checkbox('genre', label='Choose genres', options=['action','comedy','crime','drama','fantasy',\
                                                            'horror','mystery','romance','science fiction',\
                                                            'sport','thriller','war','western'], inline=True)
        pin.put_input('key_words', label='Insert keywords')
        pywebio.output.put_button('GENERATE', onclick=self.submission, color='success')
        while True:
            if self.submit:
                put_text('Generating...')
                self.submit=False
                if len(pin_obj['genre'])>3:
                    put_text('Up to 3 genres please :)')
                txt = f'<extra_id_0> {pin_obj["title"]} <extra_id_1> {",".join(pin_obj["genre"])} <extra_id_2> {pin_obj["key_words"]}'
                txt = self.p3_model.tokenizer(txt, return_tensors="pt")
                beam_outputs = self.p3_model.model.generate(
                    **txt,
                    max_length=300,
                    num_beams=10,
                    no_repeat_ngram_size=3,
                    num_return_sequences=10
                )
                put_text(self.p3_model.tokenizer.decode(beam_outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    pyweb = Pyweb()
    pywebio.start_server(pyweb.ui, port=8888,remote_access=True)