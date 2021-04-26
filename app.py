from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

model = pickle.load(open('randomforest_classifier.pkl', 'rb'))

standard_scaler = pickle.load(open('sc.pkl', 'rb'))


def prepare_data(genero, aposentado, casado, dependente, tenure, servico_telefone, multlinhas, servico_internet, seguro_online, backup_online, protecao_celular, suporte_tecnico, streamtv, streammovies, contract, paperless, payment, monthly, total):

    colunas = ['Male', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    
    is_male = 1 if genero == 'homem' else 0
    is_senior = 1 if aposentado =='sim' else 0 
    is_partner = 1 if casado == 'sim' else 0
    is_dependent = 1 if dependente == 'sim' else 0
    
    is_phoneservice = 1 if servico_telefone == 'sim' else 0
    is_multiplelines = 1 if multlinhas == 'sim' else 0
    
    if servico_internet == 'fibra óptica':
        internetservice = 1
    elif servico_internet == 'dsl':
        internetservice = 2
    elif servico_internet == 'Não':
        internetservice = 0
    
    is_onlinesecurity = 1 if seguro_online == 'sim' else 0
    is_onlinebackup = 1 if backup_online == 'sim' else 0
    is_deviceprotection = 1 if protecao_celular == 'sim' else 0
    is_techsupport = 1 if suporte_tecnico == 'sim' else 0
    is_streamingtv = 1 if streamtv == 'sim' else 0
    is_streamingmovies = 1 if streammovies == 'sim' else 0
    
    if contract == 'dois anos':
        contrato = 12
    elif contract == 'um ano':
        contrato = 12
    elif contract == 'mês':
        contrato = 1

    is_paperlessbilling = 1 if paperless == 'sim' else 0
    
    
    if payment == 'cheque eletrônico':
        payment_method = 1
    elif payment == 'cheque por correio':
        payment_method = 2
    elif payment == 'transferência bancária':
        payment_method = 3
    elif payment == 'cartão de crédito automático':
        payment_method = 4
    

    dados_entrada = [[is_male],
                 [is_senior],
                 [is_partner],
                 [is_dependent],
                [tenure],
                [is_phoneservice],
                [is_multiplelines], 
                [internetservice],
                [is_onlinesecurity],
                [is_onlinebackup],
                [is_deviceprotection],
                [is_techsupport], 
                [is_streamingtv], 
                [is_streamingmovies],
                [contrato],
                [is_paperlessbilling],
                [payment_method], 
                [monthly], 
                [total]]
                     
    dados_entrada = dict(zip(colunas, dados_entrada))
    X = pd.DataFrame(dados_entrada)
    escalados = standard_scaler.transform(X)
    final = pd.DataFrame(escalados, columns = colunas)
    return final
    
@app.route('/')
def home():
    return render_template('deploy.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = list(request.form.values())
    genero, aposentado, casado, dependente, tenure, servico_telefone, multlinhas, servico_internet, seguro_online, backup_online, protecao_celular, suporte_tecnico, streamtv, streammovies, contract, paperless, payment, monthly, total = features[0], int(features[1]), int(features[2]), int(features[3]), int(features[4]), int(features[5]), int(features[6]), int(features[7]), int(features[8]), int(features[9]), int(features[10]), int(features[11]), int(features[12]), int(features[13]), int(features[14]), int(features[15]), int(features[16]), int(features[17]), int(features[18])
    final = prepare_data(genero.lower(), aposentado.lower(), casado.lower(), dependente.lower(), tenure, servico_telefone.lower(), multlinhas.lower(), servico_internet.lower(), seguro_online.lower(), backup_online.lower(), protecao_celular.lower(), suporte_tecnico.lower(), streamtv.lower(), streammovies.lower(), contract.lower(), paperless.lower(), payment.lower(), monthly, total)
    pred = rfc.predict(final)
    churn = pred[0]
    return render_template('deploy.html', prediction_text = churn)

if __name__ == "__main__":
    app.run()
