#### Dev branch
This branch is a development branch for a new approach.
We hypothesize channel wise stacking of masks to corresponding images induce stronger spatial relations.
The separate mask encoder is removed and instead the first convolution layer of the head camera's image encoder takes 4 channels as input.