from pyspark import SparkContext
from pyspark import HiveContext
import re

file_name = 'wiki-Vote.txt'

if __name__ == "__main__":
    sc = SparkContext()
    sqlContext = HiveContext(sc)
    data = sc.textFile(file_name)

    edges = data.filter(lambda x: re.match(r'\d+\t\d+', x)).map(lambda x: map(int, re.split('\t', x)))
    edges_df = sqlContext.createDataFrame(edges, ['src', 'dst'])
    #
    vertices_all = edges.flatMap(lambda x: x).map(lambda x: (x,))
    vertices_all_df = sqlContext.createDataFrame(vertices_all, ['id'])
    vertices_df = vertices_all_df.dropDuplicates()

    # print edges_df.count()
    # print vertices_df.count()

    from graphframes import *

    g = GraphFrame(vertices_df, edges_df)

    sc.setCheckpointDir('.')

    result = g.connectedComponents()

    # print result.show()

    result.groupBy(result.component).count().show()
