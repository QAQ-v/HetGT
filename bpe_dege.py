# import sys
# import os
# from collections import defaultdict
#
# grh_path = 'train.grh'#sys.argv[1]
# amr_path = 'train.amr'#sys.argv[2]
# amr_bpe_path = 'train.amr.bpe'#sys.argv[3]
# bpe_grh_path = 'train.grh.bpe'#sys.argv[4]
#
# with open(grh_path, 'r', encoding='utf8') as f:
#     grhs = f.readlines()
#
# with open(amr_path, 'r', encoding='utf8') as f:
#     amrs = f.readlines()
#
# with open(amr_bpe_path, 'r', encoding='utf8') as f:
#     amrs_bpe = f.readlines()
#
# with open(bpe_grh_path, 'w', encoding='utf8') as f:
#     for grh, amr, amr_bpe in zip(grhs, amrs, amrs_bpe):
#         node_dict = defaultdict()
#         amr_bpe = amr_bpe.split()
#         amr = amr.split()
#         tmp = grh.rstrip().split()
#         grh = [[int(tok[1:-1].split(',')[0]),
#                  int(tok[1:-1].split(',')[1]),
#                  tok[1:-1].split(',')[2]] for tok in tmp]
#         new_grh = []
#         # tmp_grh = []
#         final_grh = []
#         word_dict = []
#         for i in range(len(amr)):
#             if amr[i] in word_dict:
#                 j = word_dict.count(amr[i])
#                 word_dict.append(amr[i])
#                 amr[i] = amr[i] + '_' + str(j)
#             else:
#                 word_dict.append(amr[i])
#         word_dict = []
#         for i in range(len(amr_bpe)):
#             if amr_bpe[i] in word_dict:
#                 j = word_dict.count(amr_bpe[i])
#                 word_dict.append(amr_bpe[i])
#                 amr_bpe[i] = amr_bpe[i] + '_' + str(j)
#             else:
#                 word_dict.append(amr_bpe[i])
#         for i in range(len(grh)):
#             new_grh.append([amr[grh[i][0]], amr[grh[i][1]], grh[i][2]])
#         FIRST = True
#         j=0
#         for i in range(len(amr_bpe)):
#             if amr[j] == amr_bpe[i]:
#                 node_dict[amr_bpe[i]] = str(i)
#                 # i += 1
#                 j += 1
#             else:
#                 if "@@" in amr_bpe[i]:
#                     node_dict[amr_bpe[i]] = str(i)
#                     new_grh.append([amr_bpe[i], amr_bpe[i+1], 'd'])
#                     if i > 0:
#                         new_grh.append([amr_bpe[i], amr_bpe[i-1], 'r'])
#                     if FIRST:
#                         for e in new_grh :
#                             if e[1] == amr[j] and e[0] != e[1] :
#                                 e[1] = amr_bpe[i]
#                             elif e[1] == amr[j] and e[0] == e[1]:
#                                 e[0], e[1] = amr_bpe[i], amr_bpe[i]
#                             else:
#                                 pass
#                         FIRST = False
#                     # i += 1
#                 else:
#                     new_grh.append([amr_bpe[i], amr_bpe[i], 's'])
#                     node_dict[amr_bpe[i]] = str(i)
#                     new_grh.append((amr_bpe[i], amr_bpe[i], 's'))
#                     for e in new_grh:
#                         if e[0] == amr[j]:
#                             e[0] = amr_bpe[i]
#                             if e[2] == 'r':
#                                 new_grh.remove([amr_bpe[i], e[1], e[2]])
#                     new_grh.append([amr_bpe[i], amr_bpe[i - 1], 'r'])
#
#                     # i += 1
#                     j += 1
#                     FIRST = True
#
#         for e in new_grh:
#             final_grh.append((node_dict[e[0]], node_dict[e[1]], e[2]))
#         sort_final_grh = sorted(final_grh)
#         for g in sort_final_grh:
#             f.write(' '.join(['('+','.join(g)+')']))
#         f.write('\n')

import sys
import os
from collections import defaultdict

grh_path = sys.argv[1]
amr_path = sys.argv[2]
amr_bpe_path = sys.argv[3]
bpe_grh_path = sys.argv[4]

# grh_path = 'train.grh'#sys.argv[1]
# amr_path = 'train.amr'#sys.argv[2]
# amr_bpe_path = 'train.amr.bpe'#sys.argv[3]
# bpe_grh_path = 'train.grh.bpe'#sys.argv[4]

with open(grh_path, 'r', encoding='utf8') as f:
    grhs = f.readlines()

with open(amr_path, 'r', encoding='utf8') as f:
    amrs = f.readlines()

with open(amr_bpe_path, 'r', encoding='utf8') as f:
    amrs_bpe = f.readlines()

with open(bpe_grh_path, 'w', encoding='utf8') as f:
    for grh, amr, amr_bpe in zip(grhs, amrs, amrs_bpe):
        node_dict = defaultdict()
        amr_bpe = amr_bpe.rstrip().split(' ')
        amr = amr.rstrip().split(' ')
        tmp = grh.rstrip().split()
        grh = [[int(tok[1:-1].split(',')[0]),
                 int(tok[1:-1].split(',')[1]),
                 tok[1:-1].split(',')[2]] for tok in tmp]
        new_grh = []
        # tmp_grh = []
        final_grh = []
        word_dict = []
        for i in range(len(amr)):
            if amr[i] in word_dict:
                j = word_dict.count(amr[i])
                word_dict.append(amr[i])
                amr[i] = amr[i] + '_' + str(j)
            else:
                word_dict.append(amr[i])
        word_dict = []
        for i in range(len(amr_bpe)):
            if amr_bpe[i] in word_dict:
                j = word_dict.count(amr_bpe[i])
                word_dict.append(amr_bpe[i])
                amr_bpe[i] = amr_bpe[i] + '_' + str(j)
            else:
                word_dict.append(amr_bpe[i])
        for i in range(len(grh)):
            new_grh.append([amr[grh[i][0]], amr[grh[i][1]], grh[i][2]])
        FIRST = True
        j=0
        for i in range(len(amr_bpe)):
            if amr[j] == amr_bpe[i]:
                node_dict[amr_bpe[i]] = str(i)
                # i += 1
                j += 1
            else:
                if "@@" in amr_bpe[i]:
                    node_dict[amr_bpe[i]] = str(i)
                    new_grh.append([amr_bpe[i], amr_bpe[i+1], 'd'])
                    new_grh.append([amr_bpe[i+1], amr_bpe[i], 'r'])
                    new_grh.append([amr_bpe[i], amr_bpe[i], 's'])
                    if FIRST:
                        indexs = []
                        for e in new_grh :
                            # if 'csubjpass' in e:
                            #     index = new_grh.index(e)
                            #     qwe = []
                            #     qwe.append(e)
                            #     q = amr[j]
                            if e[1] == amr[j] and e[0] != e[1] :
                                e[1] = amr_bpe[i]
                            elif e[1] == amr[j] and e[0] == e[1]:
                                indexs.append(e)
                            else:
                                pass
                        # indexs = []
                        # for k in range(len(new_grh)):
                        #     if new_grh[k][1] == amr[j] and new_grh[k][0] != new_grh[k][1]:
                        #         new_grh[k][1] = amr_bpe[i]
                        #     elif new_grh[k][1] == amr[j] and new_grh[k][0] == new_grh[k][1]:
                        #         indexs.append(k)
                        #     else:
                        #         pass
                        for index in indexs:
                            new_grh.remove(index)
                        FIRST = False
                    # i += 1
                else:
                    new_grh.append([amr_bpe[i], amr_bpe[i], 's'])
                    node_dict[amr_bpe[i]] = str(i)
                    for e in new_grh:
                        if e[0] == amr[j]:
                            e[0] = amr_bpe[i]
                    #         if e[2] == 'r':
                    #             new_grh.remove([amr_bpe[i], e[1], e[2]])
                    # new_grh.append([amr_bpe[i], amr_bpe[i - 1], 'r'])

                    # i += 1
                    j += 1
                    FIRST = True

        for e in new_grh:
            final_grh.append((node_dict[e[0]], node_dict[e[1]], e[2]))
        sort_final_grh = sorted(final_grh, key=lambda x:int(x[0]))
        f.write(' '.join(['('+','.join(g)+')' for g in sort_final_grh]) + '\n')
