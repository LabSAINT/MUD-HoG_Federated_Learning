from utils.allocateGPU import *

#allocate_gpu()

import parser
import _main

if __name__ == "__main__":
    args = parser.parse_args()
    _main.main(args)
