#Introduction
This tutorial uses the same technique as sentdex from youtube.
But it also uses tensorflow's built-in reader instead of manual iteration.
The reason for this is that for loops in python are known to be slow and manual iteration has to use them.
tensorflow, however, does this in C++ like all of its graph computation so theoretically, it should be much faster.
I am making this both for my own testing and because while tensorflow has many tutorial and a lot of documentation, I personally haven't found any that explains its data reading beyond feed_dict well.
