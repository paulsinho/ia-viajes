from django import forms
from django.core.validators import MinLengthValidator, MinValueValidator, MaxValueValidator


class ContactForm(forms.Form):
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea)
    sender = forms.EmailField()
    cc_myself = forms.BooleanField(required=False)

COUNTRY_CHOICES=[
    ('0', 'Seleccionar Pais'),
    ('1', 'France'),
    ('2', 'Germany'),
    ('3', 'USA'),
    ('4', 'Netherlands'),
    ('5', 'Belgium'),
    ('6', 'Irish Republic'),
    ('7', 'Australia'),
    ('8', 'Canada'),
    ('9', 'Italy'),
    ('10', 'Spain'),
    ('11', 'Switzerland'),
    ('12', 'Norway'),
    ('13', 'Japan'),
    ('14', 'Poland'),
    ('15', 'South Africa'),
    ('16', 'Denmark'),
    ('17', 'Central & South America'),
    ('18', 'Russia'),
    ('19', 'New Zealand'),
    ('20', 'Sweden'),
    ('21', 'India'),
    ('22', 'Hong Kong'),
    ('23', 'Portugsl'),
    ('24', 'Austria'),
    ('25', 'Greece'),
    ('26', 'Israel'),
    ('27', 'Middle East'),
    ('28', 'Other Africa'),
    ('29', 'Other Western Europe'),
    ('30', 'Other Asia'),
    ('31', 'Brazil'),
    ('32', 'Eastern Europe'),
    ('33', 'United Arab Emirates'),
    ('34', 'Singapore'),
    ('35', 'Czech Republic'),
    ('36', 'Malasia'),
    ('37', 'Saudi Arabia'),
    ('38', 'Nigeria'),
    ('39', 'Finland'),
    ('40', 'South Korea'),
    ('41', 'Mexico'),
    ('42', 'China'),
    ('43', 'Hungary'),
    ('44', 'Pakistan'),
    ('45', 'Turkey'),
    ('46', 'Thailand'),
    ('47', 'Egypt'),
    ('48', 'Taiwan'),
    ('49', 'Kenya'),
    ('50', 'Kuwait'),
    ('51', 'Other Eastern Europe'),
    ('52', 'Romania')]   

QUARTER_CHOICES = [
    ('0', 'Seleccionar Trimestre'),
    ('1', 'Enero-Marzo'),
    ('2', 'Abril-Junio'),
    ('3', 'Julio-Septiembre'),
    ('4', 'Octubre-Diciembre'),
]

PURPOSE_CHOICES = [
    ('0', 'Seleccionar Proposito'),
    ('1', 'Fechas Festivas'),
    ('2', 'Turismo y Lazos Sociales'),
    ('3', 'Negocios'),
    ('4', 'Turismo Variado'),
    ('5', 'Estudio')
]

MODE_CHOICES = [
    ('0', 'Seleccionar Transporte'),
    ('1', 'Aire'),
    ('2', 'Tierra'),
    ('3', 'Mar')
]

class TourismForm(forms.Form):
    year = forms.IntegerField(label="Año(ej.2020)", 
                              validators=[MinValueValidator(2000, message="Year is invalid!"), 
                              MaxValueValidator(9999, message="Ingresa 4 digitos para el año!")],)
    duration = forms.IntegerField(label="Duracion(dias)", validators=[MinValueValidator(1, message="Minimo un dia requerido!")])
    spends = forms.FloatField(label="Gastos en $", validators=[MinValueValidator(0.1, message="Gastos: free!"), MaxValueValidator(150000, message="Gastos no deben exceder 150K $")],)
    mode = forms.CharField(label="Transporte", widget=forms.Select(choices=MODE_CHOICES))
    purpose = forms.CharField(label="Proposito", widget=forms.Select(choices=PURPOSE_CHOICES))
    quarter = forms.CharField(label="Trimestre", widget=forms.Select(choices=QUARTER_CHOICES), )
    country = forms.CharField(label="Pais", widget=forms.Select(choices=COUNTRY_CHOICES))
