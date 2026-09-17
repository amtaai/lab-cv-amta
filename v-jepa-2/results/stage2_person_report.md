# Cascade Nivel 2 (deteccion de personas + seguimiento) — reporte

protocolo: yolo11 pesos=(default) imgsz=640 conf=0.35 iou=0.7 device=0 prompts=['person'] | seguimiento bytetrack.yaml min_hits=3 max_age=30

conteo: linea=[(0.35, 0.0), (0.35, 1.0)] (normalizada)

Nivel 3 (ReID): facebook/dinov2-small umbral=0.5 ventana=30.0s mascara=False

gate Nivel 1: MOG2 min_area_frac=0.0016 warmup=30 flicker_fg_ratio=0.5 | cv_threads=1

> El detector y el tracker NO son de este modulo: son `core/perception/`, los mismos que corren los notebooks de `yolo_seg/` y `yolo_world/`. El cascade los cablea al filtro de movimiento y les mide el costo.

## Por clip

| clip_id | frames | % al Nivel 2 | personas | entra | sale | conf media | ms gpu/frame |
|---|---|---|---|---|---|---|---|
| normal_normal-1 | 318 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-10 | 351 | 34.89 | 3 | 0 | 0 | 0.671 | 12.5 |
| normal_normal-11 | 350 | 87.50 | 6 | 1 | 0 | 0.756 | 10.8 |
| normal_normal-12 | 346 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-13 | 311 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-14 | 309 | 94.98 | 1 | 0 | 1 | 0.871 | 10.4 |
| normal_normal-15 | 315 | 56.84 | 3 | 1 | 0 | 0.826 | 10.7 |
| normal_normal-16 | 352 | 95.03 | 3 | 1 | 0 | 0.729 | 10.5 |
| normal_normal-17 | 351 | 69.47 | 2 | 0 | 1 | 0.844 | 10.3 |
| normal_normal-18 | 320 | 22.41 | 1 | 0 | 0 | 0.891 | 10.6 |
| normal_normal-19 | 320 | 47.24 | 1 | 0 | 0 | 0.888 | 10.4 |
| normal_normal-2 | 320 | 2.41 | 1 | 0 | 0 | 0.853 | 10.5 |
| normal_normal-20 | 350 | 48.12 | 1 | 0 | 0 | 0.859 | 10.3 |
| normal_normal-21 | 323 | 83.96 | 1 | 1 | 0 | 0.872 | 10.4 |
| normal_normal-22 | 337 | 53.09 | 1 | 1 | 0 | 0.924 | 10.5 |
| normal_normal-23 | 318 | 89.58 | 1 | 0 | 1 | 0.914 | 10.4 |
| normal_normal-24 | 320 | 47.93 | 1 | 0 | 0 | 0.873 | 10.3 |
| normal_normal-25 | 314 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-26 | 333 | 7.92 | 2 | 0 | 0 | 0.698 | 10.4 |
| normal_normal-27 | 321 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-28 | 318 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-29 | 323 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-3 | 336 | 74.18 | 2 | 0 | 2 | 0.886 | 10.3 |
| normal_normal-30 | 325 | 40.68 | 1 | 1 | 0 | 0.909 | 10.2 |
| normal_normal-31 | 361 | 71.90 | 1 | 0 | 1 | 0.904 | 10.4 |
| normal_normal-32 | 351 | 54.21 | 1 | 0 | 1 | 0.837 | 10.3 |
| normal_normal-33 | 232 | 79.21 | 1 | 0 | 1 | 0.868 | 10.4 |
| normal_normal-34 | 290 | 74.62 | 1 | 0 | 1 | 0.881 | 10.3 |
| normal_normal-35 | 329 | 89.63 | 1 | 0 | 1 | 0.901 | 10.1 |
| normal_normal-36 | 366 | 36.90 | 1 | 0 | 1 | 0.910 | 10.3 |
| normal_normal-37 | 311 | 83.63 | 1 | 0 | 1 | 0.905 | 10.4 |
| normal_normal-38 | 315 | 84.56 | 1 | 1 | 0 | 0.899 | 10.4 |
| normal_normal-39 | 312 | 0.00 | 0 | 0 | 0 | 0.000 | 0.0 |
| normal_normal-4 | 336 | 91.83 | 2 | 0 | 1 | 0.891 | 10.4 |
| normal_normal-40 | 157 | 83.46 | 6 | 0 | 1 | 0.777 | 10.9 |
| normal_normal-41 | 357 | 55.05 | 2 | 0 | 0 | 0.844 | 10.0 |
| normal_normal-42 | 339 | 63.43 | 7 | 0 | 1 | 0.710 | 9.9 |
| normal_normal-43 | 620 | 82.20 | 4 | 1 | 0 | 0.753 | 10.1 |
| normal_normal-44 | 349 | 82.45 | 1 | 0 | 0 | 0.880 | 9.8 |
| normal_normal-45 | 316 | 90.91 | 1 | 1 | 0 | 0.864 | 9.9 |
| normal_normal-46 | 347 | 48.26 | 1 | 0 | 1 | 0.864 | 9.9 |
| normal_normal-47 | 321 | 26.46 | 1 | 0 | 0 | 0.876 | 10.0 |
| normal_normal-48 | 320 | 10.69 | 1 | 0 | 0 | 0.881 | 9.8 |
| normal_normal-49 | 349 | 45.14 | 1 | 0 | 0 | 0.850 | 9.8 |
| normal_normal-5 | 336 | 38.89 | 1 | 0 | 1 | 0.927 | 9.9 |
| normal_normal-50 | 320 | 64.14 | 1 | 0 | 0 | 0.887 | 9.8 |
| normal_normal-51 | 320 | 64.14 | 1 | 0 | 0 | 0.887 | 9.9 |
| normal_normal-52 | 349 | 82.45 | 1 | 0 | 0 | 0.880 | 9.8 |
| normal_normal-53 | 321 | 34.02 | 1 | 0 | 0 | 0.879 | 9.8 |
| normal_normal-54 | 358 | 39.02 | 1 | 0 | 0 | 0.914 | 9.9 |
| normal_normal-55 | 358 | 13.11 | 1 | 0 | 0 | 0.889 | 9.9 |
| normal_normal-56 | 358 | 13.11 | 1 | 0 | 0 | 0.889 | 10.0 |
| normal_normal-57 | 336 | 73.20 | 2 | 0 | 1 | 0.897 | 10.0 |
| normal_normal-58 | 336 | 94.12 | 2 | 1 | 0 | 0.881 | 10.0 |
| normal_normal-59 | 337 | 56.35 | 1 | 1 | 0 | 0.924 | 9.9 |
| normal_normal-6 | 346 | 76.58 | 2 | 0 | 0 | 0.763 | 10.0 |
| normal_normal-60 | 305 | 54.91 | 1 | 0 | 0 | 0.914 | 10.0 |
| normal_normal-61 | 336 | 79.41 | 3 | 0 | 1 | 0.869 | 11.2 |
| normal_normal-62 | 337 | 43.32 | 1 | 0 | 1 | 0.925 | 10.9 |
| normal_normal-63 | 337 | 56.68 | 1 | 1 | 0 | 0.922 | 10.1 |
| normal_normal-64 | 337 | 76.55 | 1 | 1 | 0 | 0.923 | 10.0 |
| normal_normal-65 | 346 | 89.56 | 3 | 0 | 1 | 0.702 | 10.0 |
| normal_normal-66 | 311 | 24.20 | 2 | 0 | 0 | 0.829 | 10.2 |
| normal_normal-67 | 346 | 50.00 | 2 | 0 | 0 | 0.772 | 11.1 |
| normal_normal-68 | 380 | 51.14 | 2 | 0 | 0 | 0.823 | 10.9 |
| normal_normal-69 | 325 | 96.95 | 3 | 0 | 1 | 0.859 | 11.2 |
| normal_normal-7 | 341 | 88.42 | 2 | 0 | 1 | 0.758 | 11.9 |
| normal_normal-70 | 321 | 75.60 | 1 | 0 | 1 | 0.873 | 10.8 |
| normal_normal-71 | 341 | 41.16 | 2 | 0 | 1 | 0.753 | 12.0 |
| normal_normal-72 | 341 | 53.70 | 2 | 1 | 0 | 0.774 | 11.2 |
| normal_normal-73 | 319 | 80.62 | 3 | 0 | 1 | 0.760 | 11.2 |
| normal_normal-74 | 325 | 86.10 | 3 | 0 | 1 | 0.723 | 10.7 |
| normal_normal-75 | 350 | 94.38 | 2 | 0 | 1 | 0.753 | 10.3 |
| normal_normal-76 | 315 | 87.02 | 2 | 0 | 0 | 0.725 | 10.3 |
| normal_normal-77 | 350 | 77.19 | 2 | 0 | 0 | 0.738 | 10.3 |
| normal_normal-78 | 316 | 61.19 | 2 | 0 | 1 | 0.749 | 10.7 |
| normal_normal-79 | 351 | 70.72 | 2 | 0 | 1 | 0.716 | 10.5 |
| normal_normal-8 | 350 | 63.12 | 2 | 0 | 0 | 0.746 | 10.9 |
| normal_normal-80 | 316 | 56.99 | 3 | 0 | 1 | 0.706 | 10.7 |
| normal_normal-81 | 351 | 53.58 | 3 | 0 | 1 | 0.674 | 10.8 |
| normal_normal-82 | 351 | 76.01 | 4 | 0 | 1 | 0.712 | 10.6 |
| normal_normal-83 | 316 | 89.86 | 4 | 0 | 1 | 0.738 | 10.7 |
| normal_normal-84 | 315 | 91.23 | 5 | 0 | 0 | 0.722 | 11.1 |
| normal_normal-85 | 350 | 64.06 | 7 | 0 | 0 | 0.674 | 11.0 |
| normal_normal-86 | 350 | 99.69 | 7 | 2 | 0 | 0.792 | 10.8 |
| normal_normal-87 | 349 | 99.37 | 8 | 2 | 0 | 0.809 | 11.2 |
| normal_normal-88 | 316 | 55.59 | 1 | 0 | 0 | 0.868 | 10.8 |
| normal_normal-89 | 506 | 55.46 | 1 | 0 | 0 | 0.858 | 10.9 |
| normal_normal-9 | 315 | 82.81 | 2 | 0 | 0 | 0.724 | 11.1 |
| normal_normal-90 | 620 | 45.25 | 1 | 1 | 0 | 0.866 | 10.9 |
| retail_iprox_caja | 2576 | 91.44 | 10 | 2 | 1 | 0.770 | 10.6 |
| retail_iprox_tienda | 1452 | 89.24 | 28 | 8 | 7 | 0.616 | 11.6 |
| retail_usa_tienda | 1810 | 76.85 | 11 | 2 | 5 | 0.677 | 11.0 |
| shoplifting_shoplifting-1 | 311 | 45.55 | 3 | 0 | 1 | 0.708 | 10.8 |
| shoplifting_shoplifting-10 | 321 | 100.00 | 4 | 1 | 0 | 0.858 | 10.6 |
| shoplifting_shoplifting-11 | 344 | 38.22 | 1 | 0 | 0 | 0.911 | 11.1 |
| shoplifting_shoplifting-12 | 322 | 83.90 | 1 | 0 | 0 | 0.907 | 10.4 |
| shoplifting_shoplifting-13 | 334 | 21.05 | 1 | 0 | 0 | 0.920 | 10.0 |
| shoplifting_shoplifting-14 | 325 | 78.98 | 2 | 1 | 0 | 0.818 | 10.6 |
| shoplifting_shoplifting-15 | 312 | 47.52 | 2 | 0 | 1 | 0.851 | 11.1 |
| shoplifting_shoplifting-16 | 306 | 90.94 | 2 | 0 | 1 | 0.864 | 11.5 |
| shoplifting_shoplifting-17 | 321 | 71.13 | 1 | 0 | 1 | 0.868 | 10.8 |
| shoplifting_shoplifting-18 | 341 | 91.00 | 2 | 0 | 1 | 0.804 | 10.9 |
| shoplifting_shoplifting-19 | 305 | 69.45 | 3 | 0 | 1 | 0.716 | 10.9 |
| shoplifting_shoplifting-2 | 341 | 95.82 | 3 | 0 | 1 | 0.751 | 11.7 |
| shoplifting_shoplifting-20 | 335 | 56.07 | 4 | 0 | 1 | 0.740 | 11.1 |
| shoplifting_shoplifting-21 | 320 | 48.28 | 1 | 0 | 0 | 0.879 | 11.1 |
| shoplifting_shoplifting-22 | 326 | 65.54 | 2 | 1 | 0 | 0.878 | 10.4 |
| shoplifting_shoplifting-23 | 335 | 57.05 | 3 | 0 | 1 | 0.714 | 10.6 |
| shoplifting_shoplifting-24 | 327 | 75.42 | 3 | 0 | 1 | 0.807 | 11.3 |
| shoplifting_shoplifting-25 | 335 | 64.92 | 4 | 0 | 1 | 0.736 | 10.6 |
| shoplifting_shoplifting-26 | 320 | 67.93 | 2 | 1 | 0 | 0.825 | 10.4 |
| shoplifting_shoplifting-27 | 309 | 81.36 | 2 | 1 | 0 | 0.804 | 10.3 |
| shoplifting_shoplifting-28 | 334 | 93.42 | 2 | 1 | 0 | 0.809 | 10.4 |
| shoplifting_shoplifting-29 | 347 | 74.76 | 1 | 0 | 0 | 0.874 | 10.8 |
| shoplifting_shoplifting-3 | 335 | 54.43 | 1 | 0 | 1 | 0.895 | 10.7 |
| shoplifting_shoplifting-30 | 320 | 74.83 | 3 | 1 | 0 | 0.772 | 10.9 |
| shoplifting_shoplifting-31 | 316 | 62.24 | 1 | 1 | 0 | 0.874 | 10.2 |
| shoplifting_shoplifting-32 | 322 | 80.48 | 1 | 1 | 0 | 0.900 | 10.5 |
| shoplifting_shoplifting-33 | 327 | 64.65 | 1 | 1 | 0 | 0.906 | 11.0 |
| shoplifting_shoplifting-34 | 300 | 84.44 | 3 | 2 | 0 | 0.802 | 10.4 |
| shoplifting_shoplifting-35 | 313 | 81.27 | 1 | 1 | 0 | 0.893 | 10.6 |
| shoplifting_shoplifting-36 | 273 | 97.94 | 2 | 1 | 0 | 0.859 | 10.4 |
| shoplifting_shoplifting-37 | 329 | 87.96 | 1 | 1 | 0 | 0.875 | 11.0 |
| shoplifting_shoplifting-38 | 318 | 58.33 | 1 | 0 | 1 | 0.844 | 10.6 |
| shoplifting_shoplifting-39 | 346 | 38.92 | 1 | 0 | 1 | 0.902 | 10.6 |
| shoplifting_shoplifting-4 | 343 | 81.79 | 3 | 0 | 2 | 0.854 | 10.2 |
| shoplifting_shoplifting-40 | 324 | 48.98 | 1 | 0 | 1 | 0.891 | 10.8 |
| shoplifting_shoplifting-41 | 211 | 59.67 | 3 | 0 | 1 | 0.814 | 10.4 |
| shoplifting_shoplifting-42 | 311 | 49.82 | 2 | 0 | 1 | 0.847 | 10.5 |
| shoplifting_shoplifting-43 | 397 | 79.56 | 3 | 0 | 0 | 0.871 | 10.5 |
| shoplifting_shoplifting-44 | 357 | 59.02 | 2 | 0 | 0 | 0.886 | 10.6 |
| shoplifting_shoplifting-45 | 365 | 97.91 | 3 | 1 | 0 | 0.883 | 10.8 |
| shoplifting_shoplifting-46 | 327 | 61.62 | 3 | 0 | 0 | 0.875 | 10.4 |
| shoplifting_shoplifting-47 | 325 | 63.05 | 2 | 0 | 1 | 0.873 | 10.2 |
| shoplifting_shoplifting-48 | 353 | 88.85 | 1 | 0 | 0 | 0.900 | 10.5 |
| shoplifting_shoplifting-5 | 357 | 71.87 | 4 | 0 | 1 | 0.751 | 10.9 |
| shoplifting_shoplifting-50 | 363 | 65.47 | 1 | 0 | 1 | 0.876 | 11.0 |
| shoplifting_shoplifting-51 | 343 | 90.73 | 3 | 1 | 0 | 0.803 | 10.8 |
| shoplifting_shoplifting-52 | 326 | 59.12 | 2 | 0 | 0 | 0.842 | 10.9 |
| shoplifting_shoplifting-53 | 347 | 86.12 | 1 | 0 | 0 | 0.884 | 10.5 |
| shoplifting_shoplifting-54 | 347 | 55.21 | 1 | 0 | 0 | 0.865 | 10.1 |
| shoplifting_shoplifting-55 | 315 | 31.58 | 1 | 0 | 0 | 0.841 | 10.4 |
| shoplifting_shoplifting-56 | 347 | 82.02 | 1 | 0 | 0 | 0.891 | 10.3 |
| shoplifting_shoplifting-57 | 347 | 68.45 | 1 | 0 | 0 | 0.883 | 10.6 |
| shoplifting_shoplifting-58 | 320 | 18.97 | 1 | 0 | 0 | 0.881 | 10.5 |
| shoplifting_shoplifting-59 | 320 | 34.14 | 1 | 0 | 0 | 0.862 | 10.8 |
| shoplifting_shoplifting-6 | 334 | 70.07 | 4 | 1 | 0 | 0.799 | 11.1 |
| shoplifting_shoplifting-60 | 321 | 100.00 | 2 | 0 | 1 | 0.859 | 11.1 |
| shoplifting_shoplifting-61 | 308 | 83.45 | 1 | 1 | 0 | 0.885 | 10.6 |
| shoplifting_shoplifting-62 | 322 | 48.97 | 1 | 0 | 0 | 0.895 | 11.2 |
| shoplifting_shoplifting-63 | 323 | 66.89 | 1 | 0 | 0 | 0.918 | 10.9 |
| shoplifting_shoplifting-64 | 344 | 85.03 | 1 | 0 | 0 | 0.907 | 11.1 |
| shoplifting_shoplifting-65 | 314 | 35.21 | 1 | 0 | 0 | 0.902 | 10.4 |
| shoplifting_shoplifting-66 | 333 | 35.31 | 1 | 0 | 0 | 0.915 | 10.5 |
| shoplifting_shoplifting-67 | 296 | 38.72 | 1 | 0 | 0 | 0.912 | 10.5 |
| shoplifting_shoplifting-68 | 316 | 45.45 | 1 | 0 | 0 | 0.915 | 10.3 |
| shoplifting_shoplifting-69 | 325 | 39.66 | 1 | 0 | 0 | 0.916 | 10.3 |
| shoplifting_shoplifting-7 | 342 | 70.19 | 2 | 1 | 0 | 0.827 | 10.5 |
| shoplifting_shoplifting-70 | 326 | 26.35 | 1 | 0 | 0 | 0.910 | 10.3 |
| shoplifting_shoplifting-71 | 309 | 97.13 | 1 | 1 | 0 | 0.842 | 10.9 |
| shoplifting_shoplifting-72 | 323 | 100.00 | 2 | 1 | 0 | 0.832 | 11.8 |
| shoplifting_shoplifting-73 | 323 | 88.74 | 1 | 1 | 0 | 0.860 | 11.3 |
| shoplifting_shoplifting-74 | 325 | 54.24 | 1 | 1 | 0 | 0.884 | 10.6 |
| shoplifting_shoplifting-75 | 290 | 61.54 | 1 | 0 | 1 | 0.841 | 10.6 |
| shoplifting_shoplifting-76 | 343 | 72.20 | 2 | 1 | 0 | 0.881 | 10.5 |
| shoplifting_shoplifting-77 | 346 | 50.00 | 2 | 0 | 0 | 0.838 | 10.6 |
| shoplifting_shoplifting-78 | 346 | 43.99 | 2 | 0 | 1 | 0.753 | 10.5 |
| shoplifting_shoplifting-79 | 316 | 76.22 | 1 | 0 | 0 | 0.887 | 10.6 |
| shoplifting_shoplifting-8 | 316 | 66.08 | 1 | 0 | 0 | 0.874 | 10.4 |
| shoplifting_shoplifting-80 | 340 | 80.32 | 1 | 1 | 0 | 0.876 | 10.6 |
| shoplifting_shoplifting-81 | 321 | 46.05 | 1 | 1 | 0 | 0.882 | 10.3 |
| shoplifting_shoplifting-82 | 321 | 56.36 | 1 | 1 | 0 | 0.890 | 10.3 |
| shoplifting_shoplifting-83 | 310 | 65.00 | 1 | 1 | 0 | 0.866 | 10.3 |
| shoplifting_shoplifting-84 | 341 | 70.74 | 3 | 1 | 0 | 0.819 | 10.6 |
| shoplifting_shoplifting-85 | 341 | 78.14 | 2 | 0 | 1 | 0.812 | 10.7 |
| shoplifting_shoplifting-86 | 341 | 95.18 | 1 | 0 | 1 | 0.878 | 10.9 |
| shoplifting_shoplifting-87 | 336 | 57.52 | 3 | 0 | 1 | 0.718 | 10.8 |
| shoplifting_shoplifting-88 | 320 | 96.55 | 1 | 1 | 0 | 0.872 | 11.1 |
| shoplifting_shoplifting-89 | 404 | 94.92 | 1 | 1 | 0 | 0.877 | 10.6 |
| shoplifting_shoplifting-9 | 354 | 82.72 | 1 | 0 | 1 | 0.873 | 10.3 |
| shoplifting_shoplifting-90 | 373 | 91.25 | 1 | 1 | 0 | 0.877 | 10.7 |
| shoplifting_shoplifting-91 | 262 | 71.12 | 1 | 1 | 0 | 0.838 | 10.7 |
| shoplifting_shoplifting-92 | 373 | 99.71 | 1 | 1 | 0 | 0.858 | 10.7 |
| shoplifting_shoplifting-93 | 340 | 80.32 | 1 | 1 | 0 | 0.865 | 10.7 |

## Deteccion

- frames analizados por el Nivel 2: **39941** de 60990 utiles (65.49 %)
- frames con al menos una persona: **39874** (99.83 % de los analizados)
- cajas totales: **70249** | confianza media **0.776**

## Seguimiento

- personas unicas (tracks confirmados): **377** sumando los 185 clips
  - El tracker se reinicia en cada clip, que es lo correcto: son grabaciones distintas. Asi que este total son personas-por-clip, no humanos distintos; el mismo actor reaparece en muchos clips.
- cajas por persona: **186.3** — sin seguimiento esas 70249 cajas se contarian como 70249 personas
- la asociacion la hace **bytetrack.yaml** adentro de ultralytics; un track cuenta como persona a partir de 3 detecciones
- el Nivel 3 recupero **268 identidades** que la oclusion habia partido. Sin el, cada vez que alguien se tapa detras de una gondola y reaparece cuenta como una persona nueva.

> **Cuanto vale este numero.** El conteo no esta validado contra etiquetas: no hay ground truth de cuantas personas hay en cada clip del corpus. Los ms/frame de este reporte son mediciones; el conteo de personas es una salida del sistema sin verificar. Validarlo pide etiquetar a mano una muestra de clips.

## Conteo de personas

Las tres reglas, para que el numero se pueda auditar:

- **entra**: el centroide del track cruza la linea de conteo hacia la derecha (o hacia abajo si la linea fuera horizontal)
- **sale**: el mismo cruce en sentido contrario
- **se contabiliza**: una sola vez por `track_id`, y solo si el registro lo confirmo con 3 detecciones. Ir y venir sobre la linea no infla el numero

- entradas: **67** | salidas: **76**
- personas distintas que cruzaron la linea: **143**
- personas distintas vistas en el cuadro (crucen o no): **377**

Los dos ultimos numeros miden cosas distintas y conviene no confundirlos: el primero es trafico por un punto y el segundo es presencia. La diferencia entre ambos es gente que se movio en el cuadro sin llegar a cruzar la linea.

## Costo por etapa

- **conteo**: 185 eventos | cpu 25848.4 ms | gpu 0.0 ms | costo USD 0.00000000
- **motion_detection**: 185 eventos | cpu 550595.8 ms | gpu 0.0 ms | costo USD 0.00000000
- **person_detection**: 185 eventos | cpu 0.0 ms | gpu 425391.5 ms | costo USD 0.00000000
- **reid**: 177 eventos | cpu 0.0 ms | gpu 18524.9 ms | costo USD 0.00000000
- **tracking**: 185 eventos | cpu 667.7 ms | gpu 0.0 ms | costo USD 0.00000000

| etapa | recurso | corre sobre | ms totales | ms/frame |
|---|---|---|---|---|
| Nivel 1 · MOG2 | CPU | los 66540 frames | 550596 | 8.27 |
| Nivel 2 · yolo11 | GPU | los 39941 con movimiento | 425392 | 10.65 |
| registro de tracks | CPU | los 39941 con movimiento | 668 | 0.017 |
| conteo por linea | CPU | los 39941 con movimiento | 25848 | 0.647 |
| Nivel 3 · ReID | GPU | solo cuando aparece un ID nuevo | 18525 | 0.464 |

> `AMTA_GPU_USD_PER_HOUR` esta en 0.0, asi que **el costo en dolares del Nivel 2 es 0 por construccion**. Los milisegundos de GPU si son reales. Fijar una tarifa verificada antes de citar dolares.

## Economia de la cascada

- C1 (Nivel 1, CPU) = **8.27 ms/frame**, sobre el 100 % de los frames
- C2 (Nivel 2, GPU) = **10.65 ms/frame**, solo sobre los que pasan
- tasa de paso medida p = **0.6549**

- GPU si el detector corriera sobre todo: 709 s
- GPU en cascada: 425 s
- **ahorro de GPU: 40.0 %**

Con las etapas en recursos distintos, la cascada ya no es un intercambio de ms contra ms: el Nivel 1 gasta CPU, que sobra, para no gastar GPU, que es el recurso caro y el que limita cuantas camaras entran por maquina. El ahorro de GPU es directamente proporcional a los frames que el Nivel 1 descarta.

## Interior comercial vs el resto del corpus

| grupo | clips | frames | tasa de paso | ms gpu/frame | personas/clip | cajas por persona |
|---|---|---|---|---|---|---|
| interior comercial | 3 | 5838 | 0.8638 | 10.99 | 16.3 | 442.0 |
| resto (oficina) | 182 | 60702 | 0.6331 | 10.60 | 1.8 | 148.1 |

Los ms/frame de GPU dependen del hardware y de la resolucion, no del contenido, asi que ahi la comparacion es directa. La tasa de paso si depende de la escena, y es el numero que nunca se pudo transferir desde la oficina.

> Con 3 clips comerciales esto es un sondeo, no una medicion: sirve para ver si los ordenes de magnitud se sostienen, no para reemplazar el barrido sobre un corpus comercial de verdad.

## Cuantas camaras entran por GPU

- escena vacia : n=10 clips, p_v = 0.0106
- escena activa: n=175 clips, p_a = 0.6872

| actividad del dia | tasa de paso | ms gpu/frame | ahorro de GPU | camaras por GPU a 25 fps |
|---|---|---|---|---|
| 5 % | 0.0445 | 0.47 | 95.6 % | 84.5 |
| 10 % | 0.0783 | 0.83 | 92.2 % | 48.0 |
| 15 % | 0.1121 | 1.19 | 88.8 % | 33.5 |
| 25 % | 0.1798 | 1.91 | 82.0 % | 20.9 |
| 50 % | 0.3489 | 3.72 | 65.1 % | 10.8 |

Sin cascada entran **3.8 camaras** por GPU, sin importar si hay alguien o no. Esa es la fila contra la que hay que comparar.

## Veredicto

El Nivel 2 corre sobre el **65.49 %** de los frames (el resto lo descarta el Nivel 1) y encuentra **377 personas unicas** en 70249 cajas, con confianza media 0.776. La cascada ahorra **40.0 %** de GPU contra correr el detector siempre. Cruzaron la linea de conteo **143** personas (67 entradas, 76 salidas).

> **ALCANCE.** 182 de los 185 clips estan indexados como `location_type=other`: NO son interiores comerciales. Las cifras describen el corpus disponible, no un comercio. Los ms/frame de cada etapa si son transferibles —dependen del hardware y de la resolucion, no del contenido—; lo que no transfiere es la tasa de paso p, y para eso esta la tabla por ciclo de actividad.