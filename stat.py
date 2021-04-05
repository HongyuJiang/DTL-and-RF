
import cProfile
import pstats

prof = pstats.Stats("result.out")
prof.sort_stats('cumtime')
prof.dump_stats('output.prof')

stream = open('output.txt', 'w')
stats = pstats.Stats('output.prof', stream=stream)
stats.sort_stats('cumtime')
stats.print_stats()