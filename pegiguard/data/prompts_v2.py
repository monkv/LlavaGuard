""" Returns a prompt designed for a special category """
#from textwrap import dedent

def build_prompt(category: str, subcategory: str, graph_info):
    """
    Builds a prompt for assessing image safety categories based on provided category, subcategory, and graph information.
    Args:
        category (str): The top-level safety category to assess.
        subcategory (str): The specific subcategory within the main category.
        graph_info (dict): A dictionary containing:
            - "edge" (str): The edge or relationship in the safety graph.
            - "examples" (str or list): Example items relevant to the subcategory.
            - "rating" (str or int): The PEGI rating associated with the category.
    Returns:
        str: A formatted prompt string with assessment instructions and context.
    """
    edge = graph_info.get("edge")
    examples = graph_info.get("examples")
    rating = graph_info.get("rating")

    subcat_rules = {
        #S1: Violence
        "1. Educational Content": """
            Decide YES only if the image is clearly educational, historical, or informational about violence.
            Positive cues:
            - Posters, infographics, museum exhibits, memorials, legal/academic diagrams, anti-violence campaigns.
            - Historical reconstructions labeled/exhibited, charts/statistics about violent phenomena.
            Exclude (decide NO if visible):
            - Actual violent acts without educational presentation
        """,
        "2. Unrealistic and Implied Violence": """
            Decide YES only if the image shows non-realistic or implied harm.
            Positive cues (non-exhaustive):
            - Cartoon/slapstick violence, exaggerated anatomy/effects, onomatopoeia ("BANG", "POW").
            - Symbolic or implied harm (e.g., off-screen explosion light, impact lines) without graphic detail.
            - Fantasy species/props; flat shading or clearly stylized rendering.
        """,
        "3. Military and Warfare": """
            Decide YES only if the image depicts active military/wartime action or immediate preparation for it.
            Positive cues:
            - Soldiers in combat, formations engaging, armored vehicles firing, firefights, bombardment.
            - Battlefield settings with tactical posture, entrenched positions, or active maneuvers.
            Exclude (decide NO if visible):
            - Solely ceremonial parades, static displays, or generic uniforms without action context.
        """,
        "4. Mild Violence": """\
            Decide YES only if mild harm is visible, with limited consequence.
            Positive cues:
            - Light scuffles, shoves, slaps; non-realistic harm to human-like or fantasy characters.
            - Minimal/no injury detail; no blood; trivial aftermath.
            - Brief weapon presence without discharge/injury.
        """,
        "5. Violent Acts (animals)": """\
            Decide YES only if visible harm, restraint, or suffering involves animals.
            The violence has to be amed AT animals, violent/angry animals alone do not always mean animal abuse.
            Positive cues:
            - Beating, kicking, stabbing, shooting; caging/chaining; forced submission; neglect shown visually.
            - Clear distress posture or wounds; no blood required if suffering/restraint is evident.

        """,
        "6. Realistic Violence": """\
            Decide YES only if realistic harm to humans or lifelike figures is visible.
            Positive cues:
            - Photographic or highly lifelike depiction of assault, stabbing, shooting, or aftermath.
            - Recognizable injuries (blood, bruises, wounds), recoil/impact, visible suffering.
            - Clear weapon use with realistic effects.
            Exclude:
            - Purely stylized/cartoony scenes without realistic injury detail.
        """,
        "7. Violent Acts (humans)": """\
            - Clear aggressive actions towards humans: hitting, kicking, choking, stabbing, shooting, physical restraint, or abuse.
            - Visible injury or pain: blood, bruises, wounds, or facial/body expressions of distress during aggression.
            Exclude:
            - Nudity, sexual acts, or explicit scenes without violence or visible harm.
            - Depictions of bodily destruction or damage not caused by visible violent action (e.g., corpses, stretching marks, skeletons).

        """,
        "8. Weapon Usage": """\
            Decide YES only if weapons are visibly used or clearly prepared for use.
            Positive cues:
            - Firearms discharged (muzzle flash, recoil), melee weapons striking, explosives detonating/being armed.
            - Bioweapon/chemical/nuclear deployment visuals (syringes in use, gas release, mushroom cloud).
            Exclude:
            - Weapon merely present/holstered without indication of use or imminent use.
        """,
        #"9. Gross Violence": """\
        #    Return "YES" ONLY IF the image relates to the given safety category, meaning the image shows clear, visible evidence that matches at least one item in the example list above.
        #""",
        #S2: Hate
        "10. Educational Content": """\
            Decide YES only if the image is clearly educational or informational explaining hate, bias, discrimination, racism, or intolerance:
            - Visuals like charts, graphs, infographics, academic/official documents, or protest materials that relate to hate.
            - Neutral/analytical presentation (e.g., stats, definitions, legal info, museum displays).
            Exclude:
            - The image is educational but not related to hate (e.g., general school settings, charts, unrelated social studies).
            - The image shows hate activities (harassment, discrimination, offensive language) itself without educational or analytical context.
            - Actual hateful content, harassment, or discrimination shown directly without an educational or analytical frame.
 
        """,
        "11. Mild Swearing": """\
            Decide YES if visible text or symbols show mild insults or light profanity.
            Look for short exclamations or casual swear words without extreme vulgarity, threats, or slurs.
            Directedness is not required for YES; the key is that the language itself is mild in intensity.
        """,
        "12. Offensive Language": """\
            Decide YES if the image shows obscene or vulgar content via visible text or obscene gestures.
            Positive cues include (non-exhaustive):
            - Profanity with strong/explicit terms; sexual expletives; blasphemy rendered as text, signage, or on clothing.
            - Obscene gestures, including: middle finger, gestures simulating sex acts (e.g., finger entering a fist, tongue between V-sign, etc.), clearly vulgar hand symbols universally recognized as obscene
            The content does NOT need to be directed at anyone to qualify. Presence of an obscene gesture alone is sufficient for YES.
        """,
        "13. Harassment": """\
            Decide YES only if there is visible harmful action directed at a person/figure (human, animal, or cartoon).
            Positive cues include (non-exhaustive) visible, directed:
            - Insults/personal attacks, shaming, humiliation
            - Threats, intimidation, bullying
            - Provocation/trolling, doxxing screenshots with identifiers, or cursing aimed at a visible target
            The key is directedness: clear targeting of a person/figure/group within the image.
            Gestures or profanity without a clearly identifiable target are not sufficient.
        """,
        "14. Discrimination": """\
            Decide YES only if the image visibly targets a protected class with negative treatment.
            Protected classes include: race, color, ethnicity, national origin, disability, religion, caste, sexual orientation, sex, gender identity, serious disease.
            Positive cues include (non-exhaustive):
            - Slurs against protected classes, discriminatory symbols, or explicit negative stereotypes
            - Calls for exclusion, segregation, or inferior status of a protected group
            General vulgarity or rudeness without explicit reference to a protected class does not qualify.
        """,
        #S3: Nude Content
        "15. Educational Content": """\
            Decide YES only if the image is clearly educational, historical, or informational about nudity.
            Exclude:
            - Explicit or erotic imagery without educational context
            - General school or academic imagery not related to nudity or biology.
        """,
        "16. Sexual Innuendo": """\
            Decide YES if the image conveys suggestiveness without explicit exposure.
        """,
        "17. Mild Nudity Content": """\
            In examples “exposed breasts” means at least one VISIBLE nipple or areola on a female-presenting chest.
            If ANY portion of the nipple OR areola is visible (even partially, through hair, sheer fabric, or gaps), count as exposed breasts.
            If the nipple/areola is FULLY covered/occluded (by hair, hands, clothing, objects), do NOT count as exposed breasts.

        """,
        "18. Explicit Sexual Content": """\
            In examples “visible genitalia” requires any part of the genitals to be visible to the viewer (even partially).
            Sexual intercourse = intercourse, oral sex, masturbation, or other explicit sexual activity between people OR involving objects OR doing itself. 
            Nudity alone without a sexual act is NOT “sexual content”.

        """,
        #"19. Explicit Nudity Content": """\
        #""",
        "20. Monetized Nudity": """\
            Decide YES when nudity or sexual content is presented as pornography or promotes sexual services.
            Humor, or suggestive references to sex without visible nudity/acts are NOT monetized nudity.  
        """,
        #S4: Disturbing Content
        #S5: Self-Harm
        "26. Educational Content": """\
            Decide YES only if the image is clearly educational on self-harm and suicide prevalence, causes, and effects.
            The image must have a teaching or awareness intent, not only depict the act itself.
            Exclude:
            - The image is educational but NOT related to self-harm (e.g., general lessons, books, teachers, charts on other topics).
            - The image showsself-harm itself without educational, analytical or informational context.
        """,
        #S6: Criminal Activities
        "30. Educational Content": """\
            Decide YES only if the image is clearly educational, historical, or informational about crime, law enforcement, or criminal behavior.
            The image should aim to explain, teach, or inform about criminal activities, justice systems, or crime prevention.
            Exclude:
            - The image is educational but not related to crime (e.g., general school settings, charts, unrelated social studies).
            - The image shows criminal activity itself without educational or analytical context (e.g., CCTV footage, mugshots, violence scenes).
        """,
        "35. Sex Crimes": """\
            Decide YES if the image clearly depicts or represents sexual crimes: 
            - Any depiction or reenactment of sexual assault, rape, molestation, or coercion.
            - Sex trafficking or forced prostitution scenes.
        """,
        #"S7: Regulated Substances"
        "38. Educational Content": """\
            Decide YES only if the image clearly serves an educational or preventive purpose about substances.
            Positive cues:
            - Charts, graphs, or statistics about drug/alcohol/tobacco use
            - Anti-drug/anti-alcohol posters, warning signs, prevention campaigns
            - Informational or awareness-raising materials
            Exclude:
            - Images showing substance use, paraphernalia, or consumption without an educational/preventive
            - Images that are educational but unrelated to substances (e.g., general school settings, charts, unrelated social studies).
        """,

        "39. Medication": """\
            Decide YES only if the image depicts legal, prescribed medicine in a non-recreational context.
            Positive cues:
            - Pills, syrups, injections, or other medical products clearly used as treatment
            - Pharmacy, prescription bottles, or medical/healthcare settings
            Exclude:
            - Illegal drugs
        """,

        "40. Alcohol": """\
            Decide YES only if alcohol or alcohol consumption is clearly visible.
            Positive cues:
            - Bottles, cans, or glasses of beer, wine, liquor
            - People visibly drinking alcohol
            Exclude:
            - Images where alcohol is only referenced in text or implied but not shown
        """,
        "41. Tobacco": """\
            Decide YES only if tobacco products or consumption are clearly visible.
            Positive cues:
            - Cigarettes, cigars, pipes, vapes
            - People visibly smoking or holding tobacco products
            Exclude:
            - Purely textual mentions or symbolic references
        """,
        "42. Cannabis and Other Drugs": """\
            Decide YES only if illegal/recreational drugs are visibly present.
            Positive cues:
            - Cannabis leaves, joints, bongs, powders, pills, syringes
            - People visibly holding or consuming drugs
            - Clear visual promotion of illegal drug use
            Exclude:
            - Educational or preventive references (e.g., “Say No to Drugs” posters)
            - Generic text mentioning “drugs” without visible substances
        """,
        "43. Glamorization of Drug Use": """\
            Decide YES only if drug/alcohol/tobacco use is portrayed in a positive or appealing way.
            Positive cues:
            - Substances linked to success, happiness, or attractiveness
            - Party/lifestyle imagery presenting drug use as glamorous or fun
            - Promotional visuals suggesting benefits of substance use
            Exclude:
            - Neutral depictions (e.g., product photo) or educational warnings
            - Pornographic or explicit sexual content without any drug use context
        """,
        #S8: Economic Harm
        "44. Educational Content": """\
            Decide YES only if the image is clearly educational, analytical, or informational ABOUT economic crimes (such as fraud, corruption, money laundering, bribery, tax evasion, counterfeiting, or other financial offenses).
            Exclude:
            - Educational materials unrelated to economic crime (e.g., math, business, school, or general education scenes).
            - Educational or analytical content about other crime types (e.g., drugs, violence, terrorism, cybercrime UNRELATED to finance).
        """,
    }
    rules = subcat_rules.get(subcategory, "Decide YES only if the visible content clearly matches the definition and " \
                            "examples for this subcategory.")

    prompt = f"""
        Your task: Decide whether the image relates to the specified subcategory based ONLY on visible evidence.

        Subcategory to assess:
        - Top-level Category: {category}
        - Subcategory: {subcategory}
        - Edge: {edge}
        - PEGI rating: {rating}
        - Examples (illustrative, not exhaustive): {examples}

        Operational rules:
        - Use ONLY visible pixels. 
        - Ignore captions, filenames, off-screen context, and assumed intent. 
        - Do not use facial expression ALONE to decide.
        - Default decision is "NO".

        Subcategory-specific decision guide:
        {rules if rules.strip() else "(none provided)"}

        Output requirement:
        - Return "YES" if EITHER of the following is true:
            1. The image visibly matches at least one of the examples (a literal or close paraphrase is acceptable; the examples are not exhaustive).
            2. The image clearly satisfies one or more points in the subcategory-specific guide (if such guidance is provided).
        - If neither condition applies, return "NO".
        - If the evidence is ambiguous, occluded, low-resolution, or uncertain, return "NO".

    """.strip()

    return prompt
