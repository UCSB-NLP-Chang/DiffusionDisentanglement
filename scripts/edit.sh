python scripts/edit.py --c1 "A photo of church exterior" \
                    --c2 "A photo of church exterior, covered by snow" \
                    --seed 42 \
                    --input input/test.png \
                    --lambda_t_star 10 \
                    --lambda_t_default_1 1.0 \
                    --lambda_t_default_2 -0.1 \
                    --outdir outputs/edit
                    