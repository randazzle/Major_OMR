from flask import Flask,request,send_from_directory,render_template
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ctc_predict as predicts

app = Flask(__name__, static_url_path='')

@app.route('/img/<filename>')
def send_img(filename):
   return send_from_directory('', filename)

@app.route("/")
def root():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
      f = request.files['file']

      predicts.audio(f)
      
      return render_template('result.html')


if __name__=="__main__":
    app.run()
