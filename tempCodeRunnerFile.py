e__)


#input_data=(38,1,2,138,175,0,1,173,0,0,2,4,2)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/cancer')
def cancer ():
    return render_template('cancer.html')
@app.route("/predict_cancer", methods=['POST'])
def predict_cancer():  