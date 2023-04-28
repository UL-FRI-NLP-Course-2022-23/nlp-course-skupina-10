import argparse

import pandas as pd
from flask import Flask, redirect, render_template, request

app = Flask(__name__)
df = None
file = None


@app.get('/')
@app.get('/<id>')
def review_get(id=None):
    global df, file
    try:
        id = int(id)
    except:
        id = None

    if id == None:
        return redirect("/{}".format(0), code=302)

    if id >= df.shape[0]:
        return None, 404

    row = df.iloc[id]

    return render_template(
        './app.jinja',
        **row.to_dict(),
        current=id+1,
        total=df.shape[0]
    )


@app.post('/<id>')
def review_post(id):
    global df, file
    try:
        id = int(id)
    except:
        id = None

    if id == None:
        return redirect("/{}".format(0), code=302)

    if id >= df.shape[0]:
        return None, 404

    data = request.form
    df.at[id, 'accuracy'] = int(data.get('accuracy'))
    df.at[id, 'fluency'] = int(data.get('fluency'))
    df.at[id, 'diversity'] = int(data.get('diversity'))
    df.to_csv(file, sep=";")

    if 'prev' in data:
        return redirect("/{}".format(id-1), code=302)
    else:
        return redirect("/{}".format(id+1), code=302)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('-p', '--port', default=5000, type=int)
    parser.add_argument('-s', '--skip', default=0, type=int)
    args = parser.parse_args()

    file = args.inputfile
    df = pd.read_csv(file, sep=";")
    df['accuracy'] = df['accuracy'].astype('Int64')
    df['fluency'] = df['fluency'].astype('Int64')
    df['diversity'] = df['diversity'].astype('Int64')

    app.run(host='0.0.0.0', port=args.port)
