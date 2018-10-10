#curl --header "Content-Type: application/json"   --request POST   --data '{"immediately"}' http://localhost:5000/nextwordpredictionModel/2
from flask import Flask, request, jsonify
from flask_cors import CORS
from nt_modified import sequence_word_prediction
import pdb
app = Flask(__name__)
cors=CORS(app)
## creating model
@app.route('/nextwordprediction/<uuid>')
def callnextwordprediction():
    print('creating the model from flask server...')
    model_check = sequence_word_prediction()
    model_check.funct()
    return 'Creating similarity model with id %s' %uuid
    
@app.route('/nextwordpredictionModel/<uuid>', methods=['GET', 'POST'])

def callQuerynextwordpredictionModel(uuid):

    print('Querying the model from flask server...')

    txt_input = "immediately"
    prdsol=sequence_word_prediction()
    txt_input = request.get_data().decode("utf-8")
    # pdb.set_trace()
    resp=prdsol.funct(txt_input)
    return resp
    
if __name__ == '__main__':

    app.run(host= '0.0.0.0',debug=True)

  
