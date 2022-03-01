# A simple script to calculate BMI
from pywebio.input import input, TEXT,input_group, checkbox
from pywebio.output import put_text
import pywebio
from T5 import PlotGenerationModel

class Pyweb():
    def __init__(self):
        self.p3_model = PlotGenerationModel('/home/student/project/model1902__kw_Rake_p3', 't5-base')

    def plot_gen(self):
        info_txt = f'lets create a movie'
        info = input_group(info_txt, [
            input('Input Movie Title', name='name', type= TEXT),
            checkbox('Choose genre', name='genre', options=['action','comedy','crime','drama','fantasy',\
                                                            'horror','mystery','romance','science fiction',\
                                                            'sport','thriller','war','western'], inline=True),
            input('insert keywords', name='key_words', type=TEXT)
        ])

        if len(info['genre'])>3:
            put_text('Up to 3 genres please :)')
        txt = f'<extra_id_0> {info["name"]} <extra_id_1> {",".join(info["genre"])} <extra_id_2> {info["key_words"]}'
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
    pywebio.start_server(pyweb.plot_gen, port=8888,remote_access=True)