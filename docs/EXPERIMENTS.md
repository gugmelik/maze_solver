# Experiments

## endocot_klein4b_lora_final_maze_test

- Название эксперимента: `endocot_klein4b_lora_final_maze_test`
- Идея эксперимента: прогнать все сохранённые LoRA чекпойнты `endocot_klein4b_lora_final` на официальных EndoCoT Maze тестах `8/16/32_test` и посчитать path accuracy + prefix overlap как в `Maze/eval`.
- Бейзлайны: чекпойнты разных эпох одного рана `endocot_klein4b_lora_final`.
- Ожидаемые результаты: path accuracy на 8/16/32 должна расти с эпохами относительно ранних чекпойнтов.
- Полученные результаты:

## endocot_klein4b_base_lora

- Название эксперимента: `endocot_klein4b_base_lora`
- Идея эксперимента: обучить LoRA на undistilled bf16 `FLUX.2-klein-base-4B` (не 4-step distilled) на Maze start->solution парах, 150000 шагов; после каждой эпохи считать accuracy на официальных `8/16/32_test` с `num_inference_steps=50` и `guidance_scale=4.0`.
- Бейзлайны: validation step 0 текущего рана.
- Ожидаемые результаты: Maze test accuracy выше относительно distilled Klein LoRA при той же длине обучения.
- Полученные результаты:
