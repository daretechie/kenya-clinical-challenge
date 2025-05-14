# Data Dictionary for Kenyan Medical Vignettes

## Overview

This data dictionary describes the dataset structure for medical
vignettes used in the Kenya LLM project. The dataset contains clinical
scenarios presented to various AI language models (GPT-4.0, LLAMA,
GEMINI) and human clinicians, along with their responses and additional
metadata.

## Fields

  ---------------------------------------------------------------------------------------------
  Field Name     Data     Description    Example 1               Example 2
                 Type                                            
  -------------- -------- -------------- ----------------------- ------------------------------
  Master_Index   Number   Unique         1                       2
                          identifier for                         
                          each vignette                          

  County         String   Kenyan county  Kiambu                  Kakamega
                          where the                              
                          nurse is                               
                          working                                

  Health level   String   Type of        Sub-county Hospitals    Dispensaries and Private
                          healthcare     and Nursing Homes       Clinics
                          facility                               

  Prompt         String   Combined nurse \"I am a nurse with 10  \"I am a nurse with 2 years of
                          profile and    years of experience in  experience in General nursing
                          clinical       General nursing working working in a Dispensaries and
                          scenario,      in a Sub-county         Private Clinics in Kakamega
                          including the  Hospitals and Nursing   county in Kenya. A
                          nurse\'s       Homes in Kiambu county  three-month-old baby brought
                          introduction   in Kenya. A 28yrs old   to the facility for
                          and the        woman came to maternity immunization, which is due
                          clinical       unit with complains of  today. On exam, the baby was
                          situation with lower abdominal pain,   pale and febrile. The baby was
                          a specific     38/40 weeks VE os       crying a lot. Should I treat
                          question       closed..with no fetal   the baby first or just give
                                         heart heard. As a       the vaccine and treat later?\"
                                         midwife which is the    
                                         best mode of delivery   
                                         CS or SVD?\"            

  Nursing        String   Area of        Maternal and Child      Child Health
  Competency              nursing        Health                  
                          competency                             
                          relevant to                            
                          the scenario                           

  Clinical Panel String   Medical        OBSTETRICS AND          PAEDIATRICS
                          specialty      GYNAECOLOGY             
                          category for                           
                          the vignette                           

  Clinician      String   Response from  \"Summary: 28 year old  \"Summary: A 3 month old baby
                          a human        pregnant woman 38/40    was brought to the facility
                          clinician to   weeks with complaints   for immunization. The baby was
                          the scenario   of lower abdominal      crying a lot. On examination,
                                         pain. Vaginal           the baby was pale and febrile.
                                         examination - closed    So the question is, should I
                                         os. No fetal heart      treat the baby first or just
                                         heard. DIAGNOSIS: Non   give the vaccine and treat
                                         reassuring fetal        later? Treat the baby first,
                                         status. Q1 - Which is   then give the vaccine later.\"
                                         the best mode of        
                                         delivery: obstetric     
                                         ultrasound to confirm   
                                         fetal status. Examine   
                                         the mother to rule out  
                                         absolute indications of 
                                         a Cesarean section such 
                                         as cephalopelvic        
                                         disproportion, 2        
                                         previous scars. If no   
                                         indications of a CS,    
                                         induce labour for       
                                         spontaneous vertex      
                                         delivery\"              

  GPT4.0         String   Response       \"Given the clinical    \"In this case, the baby\'s
                          generated by   scenario, where the     condition (pale, febrile, and
                          the GPT-4.0    patient is a            crying a lot) suggests that
                          language model 28-year-old woman at 38 there may be an underlying
                                         weeks gestation         illness that requires urgent
                                         presenting with lower   attention before proceeding
                                         abdominal pain, no      with the immunization\... You
                                         fetal heart sounds      should not give the vaccine
                                         heard, and a closed     immediately without addressing
                                         cervix (as indicated by the baby\'s current health
                                         the vaginal exam), the  status. Treating the
                                         management approach     underlying cause of the fever
                                         requires careful        and pallor should be the
                                         consideration\... Based priority\...\"
                                         on the information      
                                         provided, cesarean      
                                         section (CS) would      
                                         likely be the safest    
                                         mode of delivery in     
                                         this situation\...\"    

  LLAMA          String   Response       \"Based on the          \"As a professor of primary
                          generated by   information you\'ve     healthcare in Kenya, I\'d be
                          the LLAMA      provided, I would       happy to guide you on this.
                          language model recommend a Caesarean   According to the Kenya
                                         Section (CS) as the     Expanded Programme on
                                         best mode of delivery   Immunization (KEPI)
                                         for this patient. The   guidelines, immunization
                                         fact that the fetal     should be delayed in certain
                                         heart rate is not       situations, including when the
                                         audible is a concerning child is severely ill or has a
                                         sign, and in the        high fever\...\"
                                         absence of any other    
                                         information, it\'s      
                                         prudent to assume that  
                                         the fetus may be in     
                                         distress\...\"          

  GEMINI         String   Response       \"Given the scenario: a \"You should absolutely NOT
                          generated by   28-year-old woman at    give the vaccine first. A
                          the GEMINI     38-40 weeks gestation   pale, febrile, and crying
                          language model presenting with lower   three-month-old baby is
                                         abdominal pain, a       showing signs of illness.
                                         closed cervix, and      Administering a vaccine while
                                         absent fetal heart      the baby is unwell could
                                         tones, a Cesarean       potentially worsen their
                                         Section (CS) is the     condition and even be
                                         best mode of delivery.  unsafe\...\"
                                         The absence of fetal    
                                         heart tones is a        
                                         critical finding        
                                         indicating fetal        
                                         distress or             
                                         demise\...\"            

  DDX SNOMED     String   SNOMED CT      \"289909005 \| Labor,   \"271749004 \| Acute rise of
                          diagnostic     function (observable    fever (finding)\\n61462000 \|
                          codes relevant entity) \\n871479001 \| Malaria (disorder)\\n91302008
                          to the         Uncertain presentation  \| Sepsis
                          scenario       of fetus                (disorder)\\n434021000124109
                                         (finding)\\n199112006   \| At risk for anemia
                                         \| Infections of the    (finding)\"
                                         genital tract in        
                                         pregnancy (disorder)\"  
  ---------------------------------------------------------------------------------------------

## Field Details

### Master_Index

-   **Definition**: Unique numeric identifier for each vignette in the
    > dataset

-   **Format**: string

-   **Usage**: Used to reference specific clinical scenarios

-   **Constraints**: Must be unique within the dataset

### County

-   **Definition**: Kenyan county where the nurse in the scenario is
    > working

-   **Format**: Text string

-   **Examples**: Kiambu, Kakamega

-   **Usage**: Provides geographic context for the clinical scenario

### Health level

-   **Definition**: Type of healthcare facility where the scenario takes
    > place

-   **Format**: Text string

-   **Examples**:

    -   \"Sub-county Hospitals and Nursing Homes\"

    -   \"Dispensaries and Private Clinics\"

    -   \"Health Centres\"

-   **Usage**: Indicates the level of care and resources available

### Prompt

-   **Definition**: Combined nurse profile and clinical scenario,
    > including the nurse\'s introduction and patient case

-   **Format**: Text string beginning with a standardized introduction
    > (\"I am a nurse with \[X\] years of experience\...\") followed by
    > the clinical scenario and a specific question

-   **Contents**: Nurse background, patient demographics, presenting
    > symptoms, relevant clinical findings, and a specific clinical
    > question

-   **Usage**: The complete prompt that models and clinicians respond to

### Nursing Competency

-   **Definition**: Area of the Kenya nursing practice relevant to the
    > scenario

-   **Format**: Text string

-   **Examples**:

    -   \"Maternal and Child Health\"

    -   \"Child Health\"

    -   \"Adult Health\"

-   **Usage**: Categorizes the scenario by nursing specialty area

### Clinical Panel

-   **Definition**: Medical specialty category for the expert panel
    > evaluating the Clinician and LLM responses

-   **Format**: Text string

-   **Examples**:

    -   \"OBSTETRICS AND GYNAECOLOGY\"

    -   \"PAEDIATRICS\"

    -   \"CRITICAL CARE\"

-   **Usage**: Categorizes the scenario by medical specialty for
    > evaluation purposes

### Clinician

-   **Definition**: Response from a human healthcare professional to the
    > clinical scenario

-   **Format**: Text string, often starting with summary followed by
    > assessment and recommendations

-   **Contents**: Clinical reasoning, differential diagnosis, management
    > plan, and specific answers to the scenario questions

-   **Usage**: Provides the human expert response baseline for
    > comparison with AI models

### GPT4.0, LLAMA, GEMINI

-   **Definition**: Responses generated by the GPT-4.0, LLAMA, and
    > GEMINI large language models

-   **Format**: Detailed text response

-   **Contents**: Analysis of the scenario, differential diagnosis,
    > management recommendations, and answers to the clinical questions

-   **Usage**: Used to evaluate GPT-4.0\'s clinical reasoning and
    > recommendations

### DDX SNOMED

-   **Definition**: SNOMED CT (Systematized Nomenclature of Medicine
    > Clinical Terms) codes for possible diagnoses

-   **Format**: Multiple SNOMED codes with descriptions, separated by
    > newlines

-   **Example**: \"289909005 \| Labor, function (observable entity)\"

-   **Usage**: Provides standardized medical terminology for the
    > diagnostic considerations

## Relationships Between Fields

-   **Prompt** combines the nurse\'s introduction (containing **County**
    > and **Health level** information) with the clinical scenario

-   **Prompt** contains the clinical scenario that all response fields
    > (**Clinician**, **GPT4.0**, **LLAMA**, **GEMINI**) address

-   **DDX SNOMED** provides standardized diagnostic codes that should
    > correspond to conditions mentioned in the responses

-   **Clinical Panel** and **Nursing Competency** categorize the
    > scenario by medical specialty and nursing domain

## Notes on Usage

1.  The **Prompt** field contains the complete information that should
    > be sufficient for clinical decision-making.

2.  **DDX SNOMED** codes can be used to verify if models identified the
    > key diagnostic considerations.

3.  **County** and **Health level** provide important context about
    > resource availability that may impact appropriate management
    > decisions.
