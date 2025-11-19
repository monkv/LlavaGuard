
def build_prompt(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - If the subcategory includes (humans) or (animals), judge EXCLUSIVELY wether there are safety concerns TOWARDS the given group. e.g. an angry animal is NOT always an animal abuse.
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".

        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_violence(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        General guidance for S1: Violence (do NOT classify by PEGI alone; use it only to judge realism):
        - PEGI 7: Violence is always unrealistic; often fantasy/cartoon; may include implied/off-screen harm. Very mild non-realistic violence to humans may appear.
        - PEGI 12: Mild violence; humans may appear only if the depiction looks unrealistic or leads to trivial injury (e.g., a slap). More graphic violence is allowed if targets are fantasy characters.
        - PEGI 16: More mature/realistic human violence; injury/death themes; blood/gore may appear (esp. arcade-style).
        These notes inform the **realism test** below but your decision must match the **current subcategory**.

        Core visual tests (judge ONLY visible pixels; no text/captions/off-screen inference):
        - Violence present? A deliberate act of physical harm (hitting, kicking, stabbing, shooting, bombing, etc.) or its immediate impact (impact lines, recoil, pain reaction) is visible.
        - Realism test:
        • UNREALISTIC/FANTASY cues → cartoon/flat outlines, exaggerated anatomy, slapstick effects, fantasy species, onomatopoeia, symbolic or implied harm (e.g., off-screen explosion).  
        • REALISTIC cues → photographic or highly lifelike rendering; human or human-like bodies; natural proportions/lighting/textures; recognizable injury detail (blood, bruises, wounds), lifelike weapon use, visible suffering.
        - "Educational content" means image has an educational, awareness, or informational purpose about violence towards both humans and animals (e.g., posters, infographics, anti-violence campaigns, memorials, statistics), even if no direct act of violence is depicted.
        
        Special rule for "1. Educational Content":
        - Return YES if the image serves an educational, awareness, or informational purpose about violence (e.g., posters, infographics, anti-violence campaigns, memorials, statistics), even if no direct act of violence is depicted. 
        - The violence that the image educates about can be directed towards anyone, including humans and animals.
        - The decisive factor is the intent to INFORM or RAISE AWARENESS, not the presence of harm.


        Special rule for subcategory "5. Violent Acts (animals)":
        - Return YES when animals are visibly harmed, restrained, abused, or suffering.
        - Includes: caging, chaining, beating, forced submission, neglect shown visually, or animals in clear distress.
        - Do NOT require blood/gore; suffering and forced restraint are sufficient.


        Important constraints:
        - If a subcategory specifies (humans) or (animals), judge EXCLUSIVELY harm toward that group.
        - Angry/tense poses alone is not violence.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".
        - Do NOT promote or demote based on PEGI alone; PEGI informs realism, but you must match the current subcategory.

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image matches the examples and guidence for the subcategory above (examples help but are not required if visual evidence fits).
        - Return EXACTLY "YES" or "NO".
        
        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_nudity(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Operational rules (apply literally to what is VISIBLE in the image):
        - “Exposed breasts” means at least one VISIBLE nipple or areola on a female-presenting chest.
        - If ANY portion of the nipple OR areola is visible (even partially, through hair, sheer fabric, or gaps), count as exposed breasts.
        - If the nipple/areola is FULLY covered/occluded (by hair, hands, clothing, objects), do NOT count as exposed breasts.
        - Male-presenting bare chest (visible nipples on a flat/pectoral chest) does NOT count as exposed breasts.
        - “Visible genitalia” requires any part of the genitals to be visible to the viewer (even partially).
        - Sexual intercourse = intercourse, oral sex, masturbation, or other explicit sexual activity between people OR involving objects OR doing itself. Nudity alone without a sexual act is NOT “sexual content”.

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above
        (literal match or close paraphrase).
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        
        - Return EXACTLY "YES" or "NO".
        """.strip()
    return prompt

def build_prompt_disturbing(category: str, subcategory: str, graph_info):
    """builds a prompt from a given category and subcategory"""
    #print(subcategory)
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - If the subcategory includes (humans) or (animals), judge EXCLUSIVELY wether there are safety concerns TOWARDS the given group. e.g. an angry animal is NOT always an animal abuse.
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".

        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_selfharm(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".
        
        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_criminal(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".
        
        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_violence2(category: str, subcategory: str, graph_info):

    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}
        
        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - If the subcategory includes (humans) or (animals), judge EXCLUSIVELY wether there are safety concerns TOWARDS the given group. e.g. an angry animal is NOT always an animal abuse.
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".

        Return EXACTLY either "YES" or "NO"

        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_substances(category: str, subcategory: str, graph_info):
    """is this docstring? ok it is """
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
       
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Operational rules:
        - Educational/Preventive content (warnings, anti-drug posters, “No drugs” signs, charts on drug statistics) 
        → always classify under “Educational Content (38)”.
        - Only classify as “Cannabis and Other Drugs (42)” if the image shows:
        - Actual drugs (cannabis leaves, pills, powders, syringes, etc.), OR
        - People consuming/holding drugs, OR
        - Clear visual promotion of drug use.
        - Do NOT label something as Cannabis/Other Drugs based solely on the WORD “Drugs” in text 
        if the context is prohibition, warning, or prevention.
        - “Glamorization of Drug Use (43)” applies ONLY when the content portrays drug use as positive, glamorous, fun, or beneficial.
        - “Medication (39)” applies ONLY to legal, prescribed medicine in a non-recreational context.

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "NO".
        
        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_child_explot(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" if the image relates to the given safety category, meaning the image shows evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "YES".
        
        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt

def build_prompt_hate(category: str, subcategory: str, graph_info):
    edge = graph_info["edge"]
    examples = graph_info["examples"]
    rating = graph_info["rating"]
    
    prompt = f"""
        Definition:
        - Top-level Category: {category}
        - Subcategory to assess: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples:  {examples}

        Operational rules (apply ONLY to what is visible in the image):
        - Do NOT label content as harassment, offensive language, or discrimination based on facial expressions, gestures, or emotions alone (e.g., someone looking angry or upset is NOT harassment).
        - Harassment requires visible evidence of harmful DIRECTED action:
        - Text, gestures, or actions clearly targeting **another person or group** (e.g., insults, threats, shaming).
        - Context where the aggression is aimed at someone, not just general mood.
        - Offensive Language requires **visible vulgar or offensive words** (e.g., text overlays, signs, graffiti).
        - Discrimination requires **visible targeting of protected groups** (e.g., racist slurs, derogatory imagery, discriminatory symbols).

        Assessment steps:
        - Default decision is "NO".
        - Return "YES" if the image relates to the given safety category, meaning the image shows evidence that matches at least one item in the example list above
        (literal match or close paraphrase). Though the category includes examples, but is not limited to them. 
        - Do NOT infer from mood, implication, text overlays, captions, or off-screen context.
        - If cues are uncertain/occluded/low-res/ambiguous, decide "YES".
        
        Return EXACTLY either "YES" or "NO"
        """.strip()
    return prompt