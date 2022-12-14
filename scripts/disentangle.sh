python scripts/disentangle.py --c1 "A photo of person" \
                    --c2 "A photo of person, smiling" \
                    --seed 42 \
                    --lambda_t_star 20 \
                    --lambda_t_default_1 1.0 \
                    --lambda_t_default_2 0.0 \
                    --outdir outputs/disentangle
