from src.datasets.base_dataset import BaseDataset
from src.datasets.data_utils import get_bigger_crop
from copy import deepcopy
import torch
import json


class ManValDataset(BaseDataset):
    def __init__(self, data_json_pth=None, images_path = None, embeds_path = None, *args, **kwargs):

        self.images_path = images_path
        self.id_embeds = torch.load(embeds_path)

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = {
            'nm0000609': [1, 2, 3, 5, 7], 
            'nm0000334': [0, 1, 2, 3, 5], 
            'nm0186505': [0, 1, 2, 3, 4, 7], 
            'nm0003683': [0, 3, 4, 5], 
            'nm0005452': [0, 1, 4, 5, 8], 
            'dai_yun_fan': [0, 1, 2, 3, 5, 6], 
            'jang_ki_yong': [0, 1, 2, 3, 4],  
            'nm0000563': [0, 1, 2, 4, 5, 6],
            'nm0511088': [0, 2, 3, 4, 5, 6, 9]
        }
        prompts = [
            {
                "facial_caption": "man img, mohawk purple hairstyle, screaming, looks up",
                "pose_caption": "full body view, playing red electro guitar, singing, wears black leather jacket",
                "background_caption": "on rock concert scene, blue spotlights, burning torches behind",
            },
            {
                "facial_caption": "man img with beard and sideburns, brown cowboy hat, green eyes, laughing",
                "pose_caption": "riding a horse, full body view, wears white suit over a black shirt, shows thumb up",
                "background_caption": "on a black horse with brown saddle, surounded by blue flowers, old windmill behind",
            },
            {
                "facial_caption": "man img, bald, surprised, green headband, freckles",
                "pose_caption": "surfing, full body view, spread arms to the sides, wears blue shorts and pink t-shirt",
                "background_caption": "on the wave, sunset, island with palm trees behind",
            },
            {
                "facial_caption": "man img, undercut silver hair, smiling, thin wire-frame glasses, mustache",
                "pose_caption": "full body view, conducting an orchestra, wearing a black tailcoat with crimson bow tie, left arm raised",
                "background_caption": "in a grand concert hall with gilded balconies, velvet curtains, and crystal chandeliers, dim stage lighting",
            },
            {
                "facial_caption": "man img, goatee, wry smile, freckled cheeks, black afro hairstyle",
                "pose_caption": "full body view, skateboarding down a ramp, wearing ripped denim jeans and white sneakers, mid-air trick",
                "background_caption": "in 1990s-style skate park with graffiti-covered concrete bowls and palm trees, midday haze, skyscrapers behind",
            }
        ]

        index = self.create_index(prompts, data_json)
        super().__init__(index, *args, **kwargs)

    def create_index(self, prompts, data_json):
        index = []
        for id in self.ids:
            references = []
            all_id_imges = data_json[id]
            img_nums = sorted(list(all_id_imges.keys()))
            id_embeds = []
            for img_ind in self.ids[id]:
                ref_img_num = img_nums[img_ind]

                ref_img = self.load_object(f"{self.images_path}/{id}/{ref_img_num}.jpg")
                id_embeds.append(self.id_embeds[f"{self.images_path}/{id}/{ref_img_num}.jpg"])
                ref_img_data = deepcopy(data_json[id][ref_img_num])
                ref_crop = ref_img_data['new_face_crop']
                ref_face_img = get_bigger_crop(deepcopy(ref_img), face_crop=ref_crop)
                references.append(ref_face_img)

            for prompt_num, prompt in enumerate(prompts):
                item = {}
                item["pixel_values"] = ref_img
                item["ref_images"] = deepcopy(references)
                item["prompt"] = self.get_prompt(prompt)
                item["id_embed"] = id_embeds[prompt_num % len(id_embeds)]
                item.update(prompt)

                item["original_sizes"] = (1024, 1024)
                item["crop_top_lefts"] = (0, 0)
                item["id"] = id
                index.append(item)
        return index

    def __getitem__(self, ind):
        img_dict = self._index[ind]
        instance_data = self.preprocess_data(img_dict)
        return instance_data

    def get_prompt(self, prompt):
        return ", ".join([prompt["facial_caption"], prompt["pose_caption"], prompt["background_caption"]])


class ManValFaceDataset(ManValDataset):
    def get_prompt(self, prompt):
        return prompt["facial_caption"]
    
    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        item["face_mask"] = torch.ones(32, 32, dtype=torch.bool)
        item["body_mask"] = torch.zeros(32, 32, dtype=torch.bool)
        return item
    

class WomanValDataset(ManValDataset):
    def __init__(self, data_json_pth=None, images_path = None, embeds_path = None, *args, **kwargs):

        self.images_path = images_path
        self.id_embeds = torch.load(embeds_path)

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = {
            'nm0004812':  [0, 1, 2, 3, 4, 5, 6],
            'nm0005306':  [1, 2, 3, 5, 6, 9],
            'nm0005028' : [0, 1, 2, 4, 9, 10, 11],
            'nm0001648' : [2, 5, 6, 8, 9, 11],
            'nm0019858':  [4, 6, 10, 13, 16, 18, 23],
            'nm0205626':  [0, 1, 3, 6, 8, 9, 10],
            'kim_zhang':  [0, 1, 2, 3, 4, 5, 6, 7],
            'bibi':  [0, 1, 2, 4, 5, 6, 7],
        }
        prompts = [
            {
                "facial_caption": "woman img, white hair tied up in a bun with a blue bow, purple eyes, mysterious expression, red lipstick",
                "pose_caption": "upper body view, holds a glass of red vine, wearing a vintage navy-blue satin blouse and white pearl necklace",
                "background_caption": "in a bar with shelves of bottles and big antique clock behind, low neon lighting",
            },
            {
                "facial_caption": "woman img, dark bobcut hairstyle with green clip, tired, in pain",
                "pose_caption": "full body view, sit in a boat and holding fishing rod, wearing a yellow navy raincoat over blue shirt and black rubber boots",
                "background_caption": "in a bright wooden boat, surrounded by lake with water lilies and pink flowers, forest behind",
            },
            {
                "facial_caption": "woman img, surprised, yellow cat eyes wide open, long red-colored wavy hair",
                "pose_caption": "upper body view, hold hands on cheeks, stands  in front of the cake, wears party hat and red t-short",
                "background_caption": "dark chocolate cake with candles in front, green balloons and golden fireworks behind, nighttime",
            },
            {
                "facial_caption": "woman img, hair braided into plaits, red baseball cap",
                "pose_caption": "full body view, doing pull-ups on a horizontal bar, wearing blue topik and pink shorts",
                "background_caption": "in a gym, set of dumbbells and treadmill behind",
            },
            {
                "facial_caption": "woman img, bald, purple makeup, head angled right, eyes closed",
                "pose_caption": "full body view, dancing, hands up, wears blue ball gown decorated with orange lines, wears white gloves",
                "background_caption": "in banquet hall, golden chandelier above, red carpet on the floor",
            }
        ]

        index = self.create_index(prompts, data_json)
        BaseDataset.__init__(self, index, *args, **kwargs)


class ManValFluxDataset(ManValDataset):
    def __init__(self, data_json_pth=None, images_path = None, embeds_path = None, *args, **kwargs):

        self.images_path = images_path
        self.id_embeds = torch.load(embeds_path)

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = {
            'nm0000609': [1, 2, 3, 5, 7], 
            'nm0000334': [0, 1, 2, 3, 5], 
            'nm0186505': [0, 1, 2, 3, 4, 7], 
            'nm0003683': [0, 3, 4, 5], 
            'nm0005452': [0, 1, 4, 5, 8], 
            'dai_yun_fan': [0, 1, 2, 3, 5, 6], 
            'jang_ki_yong': [0, 1, 2, 3, 4],  
            'nm0000563': [0, 1, 2, 4, 5, 6],
            'nm0511088': [0, 2, 3, 4, 5, 6, 9]
        }
        prompts = [
            {
                "facial_caption": "A photorealistic portrait of a loudly laughing man with a purple mohawk and a long golden unicorn horn on his forehead.",
                "pose_caption": "A man is wearing a white suit with a black shirt and blue tie, and his arms are spread out. A small badge with a ruby heart is pinned to the jacket. He is sitting on a throne.",
                "background_caption": "A man is in a cave at a dark oak table, behind a piles of gold. On the table is an ancient emerald cup with golden dragon on it filled with wine. Ink whith a big white feather, and a letter with a red seal are also on a table.",
            },
            {
                "facial_caption": "A bald man with large wooden hoop earrings with closed eyes and calm expression.",
                "pose_caption": "A man sits in the lotus position, bare-chested, wearing long orange trousers and no shoes. A large black dragon tattoo is on his chest. He holds a brown vase with blue stripes above his head with both hands.",
                "background_caption": "A man is on a brown woolen carpet in a temple with white brick walls. To the left behind him is a rack with swords, on the other side is a bookcase, and on the wall directly behind him hangs a large map of the world.",
            },
            {
                "facial_caption": "A man screaming angry with his mouth wide open. He wears an orange miner's helmet with a torchlight on his head. His light curly hair sticks out from under the helmet.",
                "pose_caption": "A man looks to the camera and swings an axe from behind his back, holding it with both hands. He is dressed in a blue and white checked shirt with rolled-up sleeves and grey work trousers with braces.",
                "background_caption": "A man is in the the office. To the left behind the man is a desk with a globe on it. To the right behind the man, a portrait of a cosmonaut hangs on the wall.",
            },
            {
                "facial_caption": "A photorealistic portrait of a man with slicked-back wet black hair and a thick mustache. A clear round monocle-like lens is strapped over his right eye.",
                "pose_caption": "A man is standing half-turned, wearing a dark green velvet tailcoat over a gold-and-black striped waistcoat, black gloves, and fitted black trousers. In one hand he holds a violin bow like a rapier, and in the other he carries a closed violin case.",
                "background_caption": "A man is at an abandoned train station at dusk. Near him are cracked platform tiles, a red suitcase, a mechanical departures board, rain puddles reflecting the lights, and several pigeons near a bench.",
            },
            {
                "facial_caption": "A bald man with a long turquoise beard braided into two thick braids with tiny brass bells woven into them. He has a calm slight smile.",
                "pose_caption": "A man is wearing a white fencing jacket, black cropped trousers, and soft black shoes. One foot is placed on a wooden crate. In one hand he holds an open black parasol, and in the other he carries a small brass lantern.",
                "background_caption": "A man is on a moonlit rooftop garden. Around him are potted cacti, strings of warm lights, a long wooden bench, a telescope, and a distant city skyline.",
            },

        ]

        index = self.create_index(prompts, data_json)
        BaseDataset.__init__(self, index, *args, **kwargs)

class ManValFaceFluxDataset(ManValFluxDataset):
    def get_prompt(self, prompt):
        return prompt["facial_caption"]
    
    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        item["face_mask"] = torch.ones(32, 32, dtype=torch.bool)
        item["body_mask"] = torch.zeros(32, 32, dtype=torch.bool)
        return item

class WomanValFluxDataset(WomanValDataset):
    def __init__(self, data_json_pth=None, images_path = None, embeds_path = None, *args, **kwargs):

        self.images_path = images_path
        self.id_embeds = torch.load(embeds_path)

        with open(data_json_pth) as f:
            data_json = json.load(f)

        self.ids = {
            'nm0004812':  [0, 1, 2, 3, 4, 5, 6],
            'nm0005306':  [1, 2, 3, 5, 6, 9],
            'nm0005028' : [0, 1, 2, 4, 9, 10, 11],
            'nm0001648' : [2, 5, 6, 8, 9, 11],
            'nm0019858':  [4, 6, 10, 13, 16, 18, 23],
            'nm0205626':  [0, 1, 3, 6, 8, 9, 10],
            'kim_zhang':  [0, 1, 2, 3, 4, 5, 6, 7],
            'bibi':  [0, 1, 2, 4, 5, 6, 7],
        }
        prompts = [
                {
                    "facial_caption": "A woman with ginger hair tied into a high braided ponytail, thick eyebrows. She is smiling with closed lips, and a big translucent dragonfly ornament is clipped above her left temple.",
                    "pose_caption": "A woman is kneeling on one knee, wearing a cream blouse with rolled sleeves, olive cargo trousers, and brown lace-up boots. She is holding a big brass compass in both hands and has a tan satchel hanging across her body.",
                    "background_caption": "A woman is in a dense jungle clearing. Behind her are giant fern leaves, a fallen mossy tree trunk, a canvas tent, a metal lantern hanging from a branch, and a narrow stream flowing over dark stones.",
                },
                {
                    "facial_caption": "A photorealistic portrait of a woman with a shaved head except for one long black braided strand. Her lips are pale and matte, and she is clenching her teeth in anger.",
                    "pose_caption": "A woman is sitting sideways on a simple metal chair, wearing a structured white suit, a black harness belt around the waist, and pointed white boots. One arm rests on the chair back.",
                    "background_caption": "A woman is in a minimalist concrete gallery space. Behind her are three large geometric sculptures, a reflective metal sphere on the floor, a long narrow skylight, and a single black bench against the wall.",
                },
                {
                    "facial_caption": "A woman with thick dark curly hair gathered into two large buns, bright coral lipstick. She has a joyful laughing expression and her head is slightly tilted back.",
                    "pose_caption": "A woman is mid-dance, one leg lifted and both arms extended outward. She is wearing a layered turquoise flamenco-style dress with ruffled sleeves, a wide black belt, and black dance shoes. She holds a red shawl in both hands.",
                    "background_caption": "A woman is in an old courtyard with terracotta tiles. Behind her are hanging pots with purple flowers, an iron balcony, a small stone fountain, strings of yellow lights, and a round table with oranges in a ceramic bowl.",
                },
                {
                    "facial_caption": "A photorealistic portrait of a woman with straight black hair cut in a blunt chin-length bob, icy blue contact lenses, and a fine crackled silver pattern painted over her chin. She looks downward with with an interested look.",
                    "pose_caption": "A woman is leaning over a wooden workbench, wearing a dark gray apron over a fitted white shirt, rolled-up sleeves, and black trousers. She holds a delicate mechanical figure.",
                    "background_caption": "A woman is inside a cluttered clockmaker workshop. Behind her are shelves filled with brass gears, hanging pocket watches, a large wall clock, tiny drawers with labels, and a desk lamp casting warm light.",
                },
                {
                    "facial_caption": "A woman with waist-length silver-blonde hair blowing sideways. Her mouth is slightly open, and she wears a stern windswept expression. She wears red hat.",
                    "pose_caption": "A woman is standing on the bow of a small boat with both feet planted wide apart. She is wearing a dark navy coat with brass buttons over a cream sweater, fitted trousers, and knee-high waterproof boots. She has a gold anchor badge pinned to the coat. She grips a coiled rope in one hand.",
                    "background_caption": "A woman is on a misty lake at dawn. Behind her are reeds emerging from the water, a wooden dock with tied boats, low fog rolling over the surface, distant pine trees, and a flock of birds crossing the pale sky.",
                },
        ]

        index = self.create_index(prompts, data_json)
        BaseDataset.__init__(self, index, *args, **kwargs)