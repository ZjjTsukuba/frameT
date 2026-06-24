# -*- coding: utf-8 -*-
"""
LLM-enhanced text anchors for SeAct (label-side LLM->GNN alignment, level 2 of
the alignment ladder). Each class gets a kinematic description written by an
offline LLM (Claude, 2026-06-12) focusing on WHICH body parts move and HOW —
the motion-structure information an event camera actually sees. Final anchor =
mean of (template embedding, description embedding), L2-normalized.

Run in the pytorch env:  python3 paf/make_llm_text_emb.py
"""
import json
import os
import sys

import torch
from transformers import CLIPModel, CLIPTokenizer

DESC = {
    "clap": "a person repeatedly striking both palms together in front of the chest",
    "circle": "a person moving the whole body along a circular path, turning around",
    "jumping jack": "a person jumping with legs spreading apart and arms swinging overhead repeatedly",
    "squat down": "a person bending the knees to lower the body into a squat",
    "jump squat": "a person squatting down then leaping upward explosively, repeatedly",
    "push-up": "a person in a prone plank lowering and raising the body with the arms",
    "sit down": "a person lowering the body to sit down on a chair",
    "salute": "a person raising one hand to the forehead in a military salute",
    "bend forward": "a person bending the upper body forward at the waist",
    "hurdle start": "a person crouching in a sprint start position then bursting forward",
    "long jump": "a person running up and leaping far forward with both legs",
    "nod head": "a person nodding the head up and down repeatedly",
    "walking": "a person walking at a steady pace with alternating leg swings",
    "running": "a person running fast with large alternating leg and arm swings",
    "shake head": "a person shaking the head left and right repeatedly",
    "circle head": "a person rolling the head around in slow circles",
    "circle arm": "a person swinging a straight arm in large vertical circles",
    "raise the arm": "a person lifting one arm straight up overhead",
    "side kick": "a person kicking one leg out sideways at waist height",
    "forward kick": "a person kicking one leg forward up into the air",
    "high leg lift": "a person lifting the knees high one after another while standing",
    "waving hand": "a person waving one raised hand from side to side",
    "punch straight forward": "a person punching one fist straight forward from the shoulder",
    "catch a ball": "a person reaching out with both hands to catch an incoming ball",
    "throw a ball": "a person swinging one arm overhead to throw a ball forward",
    "catch and throw a ball": "a person catching a ball and immediately throwing it back",
    "walk with a ball": "a person walking while holding a ball in the hands",
    "circle the ball around the main body": "a person passing a ball around the waist in circles",
    "circle the ball around the leg": "a person passing a ball around and between the legs in circles",
    "open and close umbrella": "a person repeatedly opening up and folding an umbrella",
    "open the computer": "a person lifting open a laptop lid on a table",
    "close the computer": "a person pushing a laptop lid down to close it",
    "use the phone": "a person holding a phone near the chest and tapping the screen",
    "put on glasses": "a person raising both hands to place glasses onto the face",
    "put off glasses": "a person taking the glasses off the face with one hand",
    "tie shoelaces": "a person crouching down and moving the fingers to tie shoelaces",
    "take a photo": "a person holding a camera up at eye level to take a photo",
    "lift the box": "a person bending down and lifting a heavy box up with both arms",
    "put down the box": "a person lowering a box from the chest down to the ground",
    "drink water": "a person raising a bottle to the mouth and tilting the head back to drink",
    "twist the bottle cap": "a person twisting the cap of a bottle with the fingers",
    "walk with an opened unbrella": "a person walking while holding an open umbrella overhead",
    "walk with a box": "a person walking while carrying a box with both arms",
    "run with a box": "a person running while carrying a box with both arms",
    "falling down": "a person suddenly collapsing and falling down to the ground",
    "vomit": "a person bending over retching with the hands near the mouth",
    "staggering": "a person staggering unsteadily with irregular swaying steps",
    "walk with stomach pain": "a person walking hunched over clutching the stomach",
    "walk with headache": "a person walking slowly with one hand pressed on the head",
    "walk with back pain": "a person walking stiffly with one hand pressed on the lower back",
    "leg injury walking": "a person limping along with one stiff injured leg",
    "hug": "two people stepping together and wrapping their arms around each other",
    "fight": "two people throwing punches at each other in a fight",
    "wave hand to each other": "two people waving raised hands at each other",
    "handshake": "two people reaching out and shaking right hands",
    "shoulder tapping": "one person tapping the other person on the shoulder",
    "clap hand": "two people clapping their hands against each other in a high five",
    "handing box": "one person handing a box over to another person",
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    classes = json.load(open(os.path.join(here, "SeAct_classes.json")))
    missing = [c for c in classes if c.strip() not in DESC]
    assert not missing, f"no description for: {missing}"

    name = "openai/clip-vit-base-patch32"
    tok = CLIPTokenizer.from_pretrained(name)
    clip = CLIPModel.from_pretrained(name).eval()

    def embed(texts):
        with torch.no_grad():
            ids = tok(texts, padding=True, truncation=True, return_tensors="pt")
            e = clip.get_text_features(**ids)
        return e / e.norm(dim=-1, keepdim=True)

    e_tmpl = embed([f"a photo of a person {c.strip()}." for c in classes])
    e_desc = embed([DESC[c.strip()] for c in classes])
    emb = (e_tmpl + e_desc) / 2
    emb = emb / emb.norm(dim=-1, keepdim=True)
    out = os.path.join(here, "seact_text_emb_llm.pt")
    torch.save({"emb": emb, "classes": classes,
                "prompt": "mean(template, LLM kinematic description)",
                "descriptions": DESC}, out)
    cos = (e_tmpl * e_desc).sum(-1)
    print(f"saved {out} {tuple(emb.shape)}; template-vs-desc cos: "
          f"min {cos.min():.3f} mean {cos.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
