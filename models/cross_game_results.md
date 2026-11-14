| method | G1 clean spellF1 / acc / n spells | G1 masked spellF1 / acc / n spells | G2 clean spellF1 / acc / n spells | G2 masked spellF1 / acc / n spells |
|---|---|---|---|---|
| majority (published-baseline style) | 0.669 / 0.833 / 415 | 0.486 / 0.706 / 500 | 0.716 / 0.822 / 418 | 0.508 / 0.696 / 505 |
| viterbi, hand-written evidence | 0.781 / 0.882 / 303 | 0.707 / 0.813 / 315 | 0.813 / 0.859 / 318 | 0.707 / 0.810 / 323 |
| learned model alone, no decoding | 0.533 / 0.948 / 684 | 0.552 / 0.872 / 548 | 0.561 / 0.934 / 661 | 0.530 / 0.910 / 598 |
| viterbi, learned evidence, penalty 5 | 0.807 / 0.858 / 343 | 0.706 / 0.793 / 344 | 0.814 / 0.838 / 336 | 0.706 / 0.792 / 341 |
| viterbi, learned evidence, penalty 10 | 0.809 / 0.875 / 342 | 0.706 / 0.797 / 339 | 0.821 / 0.838 / 330 | 0.713 / 0.806 / 333 |
| viterbi, learned evidence, penalty 20 | 0.815 / 0.876 / 326 | 0.699 / 0.795 / 330 | 0.816 / 0.838 / 328 | 0.710 / 0.805 / 333 |
| viterbi, learned evidence, penalty 40 | 0.803 / 0.882 / 318 | 0.705 / 0.793 / 319 | 0.817 / 0.845 / 316 | 0.711 / 0.816 / 316 |
| viterbi, learned evidence, penalty 80 | 0.787 / 0.887 / 310 | 0.693 / 0.798 / 304 | 0.787 / 0.845 / 308 | 0.687 / 0.826 / 306 |
| viterbi, learned evidence, penalty 160 | 0.736 / 0.873 / 300 | 0.670 / 0.822 / 293 | 0.718 / 0.836 / 301 | 0.650 / 0.811 / 295 |
| viterbi, hand+learned evidence, penalty 5 | 0.814 / 0.874 / 340 | 0.706 / 0.796 / 342 | 0.810 / 0.838 / 336 | 0.699 / 0.787 / 341 |
| viterbi, hand+learned evidence, penalty 10 | 0.820 / 0.875 / 332 | 0.712 / 0.796 / 335 | 0.814 / 0.837 / 331 | 0.710 / 0.803 / 335 |
| viterbi, hand+learned evidence, penalty 20 | 0.815 / 0.884 / 319 | 0.709 / 0.795 / 321 | 0.813 / 0.844 / 321 | 0.713 / 0.815 / 326 |
| viterbi, hand+learned evidence, penalty 40 | 0.795 / 0.888 / 314 | 0.701 / 0.802 / 313 | 0.803 / 0.847 / 311 | 0.692 / 0.816 / 313 |
| viterbi, hand+learned evidence, penalty 80 | 0.764 / 0.882 / 300 | 0.698 / 0.836 / 299 | 0.760 / 0.851 / 303 | 0.681 / 0.828 / 300 |
| viterbi, hand+learned evidence, penalty 160 | 0.713 / 0.882 / 294 | 0.655 / 0.807 / 290 | 0.705 / 0.836 / 295 | 0.644 / 0.800 / 294 |

Proxy-GT spells (the scorer's count for the same frames): G1 clean: 302, G1 masked: 302, G2 clean: 307, G2 masked: 307
