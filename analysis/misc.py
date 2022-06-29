from IPython.display import HTML
from datetime import datetime


def make_notebook_printable(signature=None, finalize=True):
    if finalize:
        if signature is not None:
            signature = f'<br>{signature}<br>{datetime.now()}'
        else:
            signature = ''

        return HTML('''<script>
        code_show=true; 
        function code_toggle() {
         if (code_show){
         $('div.input').hide();
         $("div[class='prompt output_prompt']").css('visibility','hidden');

         } else {
         $('div.input').show();
         $("div[class='prompt output_prompt']").css('visibility','visible');
         }
         code_show = !code_show
        } 
        $( document ).ready(code_toggle);
        </script>
        <a href="javascript:code_toggle()">Automatically generated report''' + signature + '''</a>.''')
    else:
        return None
