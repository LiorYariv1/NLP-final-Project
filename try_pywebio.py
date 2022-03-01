# A simple script to calculate BMI
from pywebio.input import input, TEXT,input_group, checkbox
from pywebio.output import put_text, put_row,put_button, put_markdown
from pywebio import pin
from pywebio.pin import pin as pin_obj
import pywebio
from functools import partial
from T5 import PlotGenerationModel

class Pyweb():
    def __init__(self):
        self.p3_model = PlotGenerationModel('/home/student/project/model1902__kw_Rake_p3', 't5-base')
        self.submit = False
        self.out_scopes = []
        self.scope_number=0
        self.scope = 'scope_'
        self.max_position = 2
        self.inputs = ['title','key_words']
        self.generate =  {"label": 'GENERATE', 'value':'GENERATE', 'color': 'green'}
        self.clear_all = {'label': 'clear all output', 'value':'clear all output',"type": 'reset', 'color': 'red'}

    def submission(self):
        self.submit=True

    def clear_scopes(self, scopes):
        for scope in scopes:
            pywebio.output.clear(scope)
            self.out_scopes.remove(scope)

    def clear_widget(self):
        for input in self.inputs:
            pywebio.pin.pin_update(input, value=' ')
        pywebio.pin.pin_update('genre', value=[])

    def ui(self):
        pywebio.output.put_html("<h1>Let's Create A Movie.</h1>", sanitize=False, scope=None, position=- 1)
        pywebio.output.put_scope('input', position=1)
        with pywebio.output.use_scope('input'):
        # pywebio.output.put_markdown(r""" <h1>Let's Create A Movie.</h1>""")
            put_row(pin.put_input('title', label='Choose your Movie Title'))
            put_row(pin.put_checkbox('genre', label='Choose genres', options=['action','comedy','crime','drama','fantasy',\
                                                                'horror','mystery','romance','science fiction',\
                                                                'sport','thriller','war','western'], inline=True))
            put_row(pin.put_input('key_words', label='Insert keywords'))
            put_row([put_button('GENERATE', onclick = self.submission), None,
                     put_button('clear input', onclick = self.clear_widget, small=True, color='secondary')], size='20% 70% 10%')
            # pywebio.output.put_buttons([self.generate,self.clear_all], onclick=[self.submission,''])
            put_markdown('<br>')
        pywebio.output.put_scope('clear', position = 10)
        with pywebio.output.use_scope('clear'):
            put_button('clear all outputs', onclick=partial(self.clear_scopes,self.out_scopes), color='secondary')


        while True:
            if self.submit:
                self.submit=False
                if len(pin_obj['genre'])>3:
                    pywebio.output.popup('Up to 3 genres please :) \n try again')
                    continue
                if len(self.out_scopes)>2:
                    pywebio.output.popup('too many movies displayed <br> please clear some (or all)')
                    continue
                cur_scope = f'{self.scope_number}'
                pywebio.output.put_scope(cur_scope)
                self.out_scopes.append(cur_scope)
                self.scope_number+=1
                with pywebio.output.use_scope(cur_scope):
                    title, genre, kw = pin_obj['title'], pin_obj['genre'], pin_obj['key_words']
                    put_row([put_text (f'Movie Title: {title}')])
                    genres_txt = 'chosen genre' if len(genre)==1 else 'chosen genres: '
                    put_row([put_text (f'{genres_txt}: {genre}'),put_text(f'key words: {kw}')])
                    txt = f'<extra_id_0> {pin_obj["title"]} <extra_id_1> {",".join(pin_obj["genre"])} <extra_id_2> {pin_obj["key_words"]}'
                    with pywebio.output.use_scope('generating'+cur_scope):
                        put_text('Generating...')
                    txt = self.p3_model.tokenizer(txt, return_tensors="pt")
                    beam_outputs = self.p3_model.model.generate(
                        **txt,
                        max_length=300,
                        num_beams=10,
                        no_repeat_ngram_size=3,
                        num_return_sequences=10
                    )
                    pywebio.output.clear('generating')
                    put_text(self.p3_model.tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
                    put_row([None,put_button('clear this result', onclick = partial(self.clear_scopes,[cur_scope]), small=True,color='light')], size='90% 10%')


if __name__ == '__main__':
    pyweb = Pyweb()
    pywebio.start_server(pyweb.ui, port=8888,remote_access=True)