from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired


class SelfHarmClassifyForm(FlaskForm):
    # GPT-3 parameters
    model = SelectField(
        'Select model', 
        choices=[
            ('Zero-shot', 'Zero-shot'), 
            ('One/Few-shot', 'One/Few-shot'), 
            ('Fine-tune', 'Fine-tune')
        ], 
        validators=[DataRequired()]
    )
    temperature = FloatField('Temperature', default=0)
    max_length = IntegerField('Max length', default=6)
    top_p = FloatField('Top P', default=1)
    frequency_penalty = FloatField('Frequency penalty', default=0)
    presence_penalty = FloatField('Presence penalty', default=0)
    best_of = IntegerField('Best of', default=1)

    # GPT-2 parameters
    num_beams= IntegerField('Number of beams', default=1)
    diversity_penalty = FloatField('Diversity penalty', default=0)
    repetition_penalty = FloatField('Repetition penalty', default=0)
    length_penalty = FloatField('Length penalty', default=0)

    # Select Large Language Model type
    model_type = SelectField(
        'Select model type', 
        choices=[
            ('GPT-3', 'GPT-3'), 
            ('GPT-2', 'GPT-2')
        ], 
        validators=[DataRequired()]
    )

    # Prompt designs
    prompt = TextAreaField('Prompt', default="Here's a text and below it I will classify it as being 'non-suicide' or 'suicide' text. The text is 'suicide' text if it expresses suicidal thoughts or includes potential suicidal actions. The text is 'non-suicide' text if it formally discusses suicide or refers to other's suicide. The text is 'non-suicide' text if it is not relevant to suicide.")
    input_text = TextAreaField('Input text', validators=[DataRequired()])
    submit = SubmitField('Provide Input')