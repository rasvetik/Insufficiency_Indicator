from setuptools import setup, find_packages

setup(
    install_requires=['pandas', 'opencv_python', 'opencv_contrib_python', 'matplotlib', 'sklearn', 'datetime', 'plotly',
                      'seaborn', 'wget', 'folium', 'cython', 'convertdate', 'requests', 'bs4', 'openpyxl', 'lxml',
                      'numpy'],
    name='Ratio of death rate to recovery rate as a novel indicator for COVID-19 related health system overload',
    version='3.0.0',
    description='ANALYSIS OF COVID-19 CORONAVIRUS PANDEMIC',
    long_description='The COVID-19 pandemic has produced an unprecedented level of stress on health systems worldwide.'
                     'Rapid influx of patients in acute respiratory distress, lack of effective treatments '
                     'and the health risk to healthcare providers may be overwhelming. '
                     'Entire health systems have faced overload that threatens their integrity and function. '
                     'Finding early indicators of system overload to avoid system collapse.'
                     'The Israeli health systems parameters were investigated from March 2020 to May 2021 '
                     'in terms of the variables of the SIRD model. '
                     'We identified a novel indicator of health system overload, the ratio of the COVID-19-related '
                     'death rate to the recovery rate. Conclusion: We suggest that the novel indicator may alert'
                     'the health systems to impending collapse and approaching pandemic waves.',
    url='https://github.com/rasvetik/Insufficiency_Indicator',
    author='Sveta Raboy & Doron Bar',
    author_email='',
    license='MIT',
    classifiers=[],
    keywords={'COVID-19', 'CORONAVIRUS', 'SIR', 'SIRD', 'pandemic indicators', 'death rate', 'recovered rate'},
    setup_requires=[],
    package_data={},
    packages=find_packages(),
    python_requires='>=3.5.0',
)
