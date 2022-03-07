# A simple script to calculate BMI
from pywebio.input import input, TEXT,input_group, checkbox
from pywebio.output import put_text, put_row,put_button, put_markdown, put_column, put_html
from pywebio import pin
from pywebio.pin import pin as pin_obj
import pywebio
from functools import partial
from T5 import PlotGenerationModel
import csv
from datetime import datetime
import pandas as pd
import time
from PIL import Image


class Pyweb():
    def __init__(self):
        self.p3_model = PlotGenerationModel('model0303__kw_Rake_p3', 't5-base', num_beams=9)
        # self.submit = False
        self.out_scopes = []
        self.scope_number=0
        self.scope = 'scope_'
        self.test_df = pd.read_csv('/home/student/project/data/full_model_beams.csv')[['Title','new_genres','kw_Rake_p3','full_model_9_beams']]
        self.max_position = 3
        self.num_kw = 8
        self.inputs = ['title']+ [f'kw_{i}' for i in range(self.num_kw)]
        self.clear_all = {'label': 'clear all output', 'value':'clear all output',"type": 'reset', 'color': 'red'}
        self.ranked_scopes ={}
        self.kw_nums = {1:'st',2:'nd',3:'rd',4:'th',5:'th',6:'th',7:'th',8:'th'}

    # def submission(self):
    #     self.submit=True

    def clear_scopes(self, scopes):
        scopes = scopes.copy()
        for scope in scopes:
            pywebio.output.clear(scope)
            self.out_scopes.remove(scope)
        if len(self.out_scopes)==0:
            self.max_position=2

    def clear_widget(self):
        for input in self.inputs:
            pywebio.pin.pin_update(input, value='')
        pywebio.pin.pin_update('genre', value=[])

    def ui(self):
        # put_row([
        pywebio.output.put_html("<h1>Let's Create Movies 🎬</h1>",
                                scope=None, position=- 1)
            # pywebio.output.put_image(Image.open('/home/student/project/film_img.png'), height='10%')
        # ])
        pywebio.output.put_scope('open', position=2)
        with pywebio.output.use_scope('open'):
            put_html('<h4> This system will help you to create new movies.<br> Choose the movie title and 1-3 genres.'
                  ' Add a few keywords to guide the generator. </h4>')
        pywebio.output.put_scope('input', position=3)
        with pywebio.output.use_scope('input'):
        # pywebio.output.put_markdown(r""" <h1>Let's Create A Movie.</h1>"""
            put_row(
                pin.put_input('name', label="First, choose a Nickname", placeholder='John Doe', value=None),
                size='200px')
            put_row(
                [put_column(pin.put_input('title', label="Choose your Movie Title")),
                 None], size='55% 45%')
        # put_row(pin.put_input('title', label='Choose your Movie Title'))
            put_row(pin.put_checkbox('genre', label='Choose genres', options=['action','comedy','crime','drama','fantasy',\
                                                                'horror','mystery','romance','science fiction',\
                                                                'sport','thriller','war','western'], inline=True))
            put_column([put_row(put_markdown('Insert A Few Keywords (you dont have to use all the boxes)')),
                put_row([pin.put_input(f'kw_{i}') for i in range(self.num_kw)])], size='30% 70%')
            # put_row([put_button('GENERATE', onclick = self.submission),
            put_row([put_button('GENERATE', onclick=self.generate),
                 put_button('Random', onclick = self.test_plots, color='success'),#, help='we will randomly choose real movie data and generate a new plot'), None,
                     None, put_button('clear input', onclick = self.clear_widget, small=True, color='secondary')], size='15% 10% 65% 10%')
            put_row([
            put_markdown('<p style="background-color: #f0e6ff;"> <b> Please Rate The Results,'
                         ' It Would Help Us Improve (And Get a Good Grade 😊) </b> </p>'),
                None], size= "70% 30%")
            # pywebio.output.put_buttons([self.generate,self.clear_all], onclick=[self.submission,''])
            put_markdown('<br>')
        pywebio.output.put_scope('clear', position =1000)
        with pywebio.output.use_scope('clear'):
            put_markdown('<b> Tip </b>: try to change only one input and see how the result changes')
            # put_button('clear all outputs', onclick=partial(self.clear_scopes,self.out_scopes), color='secondary')


    def test_plots(self):
        cur_scope = f'{self.scope_number}'
        pywebio.output.put_scope(cur_scope, position = self.max_position)
        # self.max_position+=1
        self.out_scopes.append(cur_scope)
        self.ranked_scopes[cur_scope]=False
        self.scope_number+=1
        sample = self.test_df.sample(1)
        # 'kw_Rake_p3','full_model_9_beams'
        with pywebio.output.use_scope(cur_scope):
            title, genre, kw = sample['Title'].item(), sample['new_genres'].item(), sample['kw_Rake_p3'].item()
            genres_txt = 'chosen genre' if len(genre)==1 else 'chosen genres: '
            put_row([put_markdown(f'<b> Movie Title</b>: {title}  <br> <b>{genres_txt} </b> {genre} <br><b>key words </b>: {kw}')]).style('background-color: #ccf5ff')
            with pywebio.output.use_scope('generating' + cur_scope):
                put_text('Generating...')
                time.sleep(3)
            pywebio.output.clear('generating' + cur_scope)
            res = sample['full_model_9_beams'].item().replace('title2008-10-08workNY Times','')
            with pywebio.output.use_scope(cur_scope):
                put_markdown(f'<b> Your Generated Movie Plot </b> <br> {res}').style('background-color: #e6faff')
            with pywebio.output.use_scope('rate'+cur_scope):
                put_column(
                    [put_row([pin.put_radio(f'overall_rating_{cur_scope}', label='Rate the overall quality of the plot', options=[1,2,3,4,5], inline=True),
                         pin.put_radio(f'coherent_rating_{cur_scope}', label='Rate the plot coherence', options=[1, 2, 3, 4, 5],
                                       inline=True),
                         pin.put_radio(f'logic_rating_{cur_scope}', label='Rate the plot logic', options=[1, 2, 3, 4, 5],
                                       inline=True)]),
                     put_row([pin.put_input(f'comment_{cur_scope}',value=None, placeholder='Please let us know if you have more thoughts (optional)'), None,
                         put_button('Rate', onclick=partial(self.submit, title, genre, kw, cur_scope, res, True),
                                                color='info')], size='90% 2% 7%')]).style('background-color: #f7fdff;')

    def generate(self):
        print('generating')
        if pin_obj['title'] in ['', ' '] :
            pywebio.output.popup(title ='Forgot Something?', content = 'Please enter a title')
            return
        if len(pin_obj['genre'])==0:
            pywebio.output.popup(title ='Forgot Something?', content = 'Please choose at least 1 genre')
            return
        if len(pin_obj['genre'])>3:
            pywebio.output.popup(title = 'too many genres', content = 'Up to 3 genres please :) \ntry again')
            return
        if not pin_obj['title'].isascii():
            pywebio.output.popup(title ='We only understand english', content = 'your title has non-english charecters')
            return
        for i in range(self.num_kw):
            if not pin_obj[f'kw_{i}'].isascii():
                pywebio.output.popup(title='We only understand english', content=f'your {i+1}{self.kw_nums[i+1]} keyword has non-english charecters')
                return
        cur_scope = f'{self.scope_number}'
        pywebio.output.put_scope(cur_scope, position = self.max_position)
        # self.max_position+=1
        self.out_scopes.append(cur_scope)
        self.ranked_scopes[cur_scope]=False
        self.scope_number+=1
        with pywebio.output.use_scope(cur_scope):
            title, genre = pin_obj['title'], ', '.join(pin_obj['genre'])
            genres_txt = 'chosen genre' if len(pin_obj['genre'])==1 else 'chosen genres: '
            kw = [str(pin_obj[f'kw_{i}']) if pin_obj[f'kw_{i}'] not in ['',' '] else '^' for i in range(self.num_kw)]
            kw = ', '.join(kw).replace(', ^', '').replace('^, ','')
            output_txt = f'<b> Movie Title</b>: {title} &emsp; <b>{genres_txt} </b>: {genre} &emsp;  <b>key words </b>: {kw}'
            if kw=='^':
                kw = ', '.join(title.split(' '))
                put_row([put_markdown(
                    f'<b> Movie Title</b>: {title}  <br> <b>{genres_txt} </b>: {genre}')]
                ).style('background-color: #ccf5ff')
            else:
                put_row([put_markdown(
                    f'<b> Movie Title</b>: {title}  <br> <b>{genres_txt} </b>: {genre} <br><b>key words</b>: {kw}')]).style(
                    'background-color: #ccf5ff')
            # put_row([put_markdown(f'<b> Movie Title</b>: {title}'), put_markdown(f'<b>{genres_txt} </b>: {",".join(genre)}'),
            #             put_markdown(f'<b>key words </b>: {kw}')]).style('background-color: #ccf5ff')
            txt = f'<extra_id_0> {title} <extra_id_1> {genre} <extra_id_2> {kw}'
            with pywebio.output.use_scope('generating'+cur_scope):
                put_text('Generating...')
            res = self.p3_model.generate_plot(txt)
            pywebio.output.clear('generating'+cur_scope)
            put_markdown(f'<b> Your Generated Movie Plot </b> <br> {res}').style('background-color: #e6faff')
            with open(r'results.csv', 'a', newline='') as res_file:
                dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                fieldnames = ['time', 'title', 'genre', 'kw', 'gen_plot']
                writer = csv.DictWriter(res_file, fieldnames=fieldnames)
                writer.writerow({'time': dt_string, 'title': title, 'genre':genre, 'kw': kw, 'gen_plot':res})
            with pywebio.output.use_scope('rate'+cur_scope):
                put_column(
                    [put_row([pin.put_radio(f'overall_rating_{cur_scope}', label='Rate the overall quality of the plot', options=[1,2,3,4,5], inline=True),
                         pin.put_radio(f'coherent_rating_{cur_scope}', label='Rate the plot coherence', options=[1, 2, 3, 4, 5],
                                       inline=True),
                         pin.put_radio(f'logic_rating_{cur_scope}', label='Rate the plot logic', options=[1, 2, 3, 4, 5],
                                       inline=True)]),
                     put_row([pin.put_input(f'comment_{cur_scope}',value=None, placeholder='Please let us know if you have more thoughts (optional)'), None,
                         put_button('Rate', onclick=partial(self.submit, title, genre, kw, cur_scope,res,False),
                                                color='info')], size='90% 2% 7%')]).style('background-color: #f7fdff;')
                    # put_column([None,
                    #             put_button('Rate', onclick=partial(self.submit, title,genre,kw, cur_scope),color='info')],
                    #            size='30% 70%').style('align-item: bottom;')],
                    #     size = '31% 31% 31% 7%').style('background-color: #f7fdff;')
                        # put_button('clear result', onclick=partial(self.clear_scopes, [cur_scope]), small=True,
                        #             color='light')])

    def submit(self, title, genre, kw, cur_scope, res,random):
        dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        ranking = {'time': dt_string,'name':pin_obj['name'],'random':random, 'title': title, 'genre':genre, 'kw': kw, 'gen_plot':res,
                   'overall': pin_obj[f'overall_rating_{cur_scope}'],
                   'coherent': pin_obj[f'coherent_rating_{cur_scope}'],
                   'logic': pin_obj[f'logic_rating_{cur_scope}'],
                   'comment': pin_obj[f'comment_{cur_scope}']
                   }
        if ranking['overall'] is None or ranking['coherent'] is None or ranking['logic'] is None:
            pywebio.output.popup(title='Please Rate Your Results',
                                 content='we want to make our generator better 🙃')
            return
        print(ranking)
        pywebio.output.clear('rate' + cur_scope)
        put_row([None, put_button('clear result', onclick=partial(self.clear_scopes, [cur_scope]), small=True,
                    color='light')],scope=cur_scope, size="88% 12%")

        with open(r'ranking.csv', 'a', newline='') as csvfile:
            fieldnames = ['time','name','random','title', 'genre','kw','gen_plot','overall','coherent','logic','comment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(ranking)


if __name__ == '__main__':
    pyweb = Pyweb()
    pywebio.start_server(pyweb.ui, port=8080,remote_access=True)