from flask import send_from_directory, render_template, url_for, request, redirect, session, jsonify, make_response
from app import webapp
from app.forms import SelfHarmClassifyForm
from app.models import Classification, GPT2Classification

import requests


@webapp.route('/')
def main():
    return render_template("pages/main.html")

@webapp.route('/self_harm_classification', methods=['GET', 'POST'])
def self_harm_classification():

    form = SelfHarmClassifyForm(model='Fine-tune', model_type='GPT-3')
    if form.validate_on_submit():
        # Common model parameters
        model_type = dict(form.model_type.choices).get(form.model_type.data)
        model = dict(form.model.choices).get(form.model.data)
        temperature = form.temperature.data
        max_length = form.max_length.data
        top_p = form.top_p.data

        print(f"model_type: {model_type}")
        print(f"model: {model}")
        print(f"temperature: {temperature}")
        print(f"max_length: {max_length}")
        print(f"top_p: {top_p}")

        prompt = form.prompt.data
        input_text = form.input_text.data
        prompt = prompt.strip()
        input_text = input_text.strip()

        print(f"prompt: {prompt}")
        print(f"input_text: {input_text}")

        if model_type == "GPT-3":
            frequency_penalty = form.frequency_penalty.data
            presence_penalty = form.presence_penalty.data
            best_of = form.best_of.data

            print(f"frequency_penalty: {frequency_penalty}")
            print(f"presence_penalty: {presence_penalty}")
            print(f"best_of: {best_of}")

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
        else:
            num_beams = form.num_beams.data
            diversity_penalty = form.diversity_penalty.data
            repetition_penalty = form.repetition_penalty.data
            length_penalty = form.length_penalty.data

            print(f"num_beams: {num_beams}")
            print(f"diversity_penalty: {diversity_penalty}")
            print(f"repetition_penalty: {repetition_penalty}")
            print(f"length_penalty: {length_penalty}")

            classifier = GPT2Classification(
                model=model,
                temperature=temperature, 
                max_length=max_length, 
                top_p=top_p, 
                num_beams=num_beams,
                diversity_penalty=diversity_penalty,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                prompt=prompt
            )

        text, prob = classifier.ask(input_text)

        return render_template("pages/self_harm_classification.html", form=form, input_text=input_text, output_text=text, output_prob=prob)

    return render_template("pages/self_harm_classification.html", form=form)
