from flask import send_from_directory, render_template, url_for, request, redirect, session, jsonify, make_response
from app import webapp
from app.forms import SelfHarmClassifyForm
from app.models import Classification

import requests


@webapp.route('/')
def main():
    return render_template("pages/main.html")

@webapp.route('/self_harm_classification', methods=['GET', 'POST'])
def self_harm_classification():

    form = SelfHarmClassifyForm(model='Zero-shot')
    if form.validate_on_submit():
        model = dict(form.model.choices).get(form.model.data)
        temperature = form.temperature.data
        max_length = form.max_length.data
        top_p = form.top_p.data
        frequency_penalty = form.frequency_penalty.data
        presence_penalty = form.presence_penalty.data
        best_of = form.best_of.data
        prompt = form.prompt.data
        input_text = form.input_text.data

        prompt = prompt.strip()
        input_text = input_text.strip()

        print(f"model: {model}")
        print(f"temperature: {temperature}")
        print(f"max_length: {max_length}")
        print(f"top_p: {top_p}")
        print(f"frequency_penalty: {frequency_penalty}")
        print(f"presence_penalty: {presence_penalty}")
        print(f"best_of: {best_of}")
        print(f"prompt: {prompt}")
        print(f"input_text: {input_text}")

        classifier = Classification(
            model=model,
            temperature=temperature, 
            max_length=max_length, 
            top_p=top_p, 
            frequency_penalty=frequency_penalty, 
            presence_penalty=presence_penalty, 
            best_of=best_of,
            prompt=prompt
        )

        text, prob = classifier.ask(input_text)

        return render_template("pages/self_harm_classification.html", form=form, input_text=input_text, output_text=text, output_prob=prob)

    return render_template("pages/self_harm_classification.html", form=form)
