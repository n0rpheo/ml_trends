import os
import pickle

path_to_db = "/media/norpheo/mySQL/db/ssorc"


with open(os.path.join(path_to_db, "untitled", "journals.pickle"), 'rb') as handle:
    journal_dict = pickle.load(handle)
with open(os.path.join(path_to_db, "untitled", "venues.pickle"), 'rb') as handle:
    venue_dict = pickle.load(handle)

check_venues = ["icml",
                "nips",
                "aaai",
                "journal of machine learning research",
                "ijcai",
                "machine learning",
                "cikm",
                "ai magazine",
                "icdm",
                "kdd",
                "uai",
                "cvpr",
                "iclr",
                "wsdm",
                "aistats"]

check_journals = ["neurocomputing",
                  "machine learning",
                  "journal of machine learning research",
                  "neural networks : the official journal of the international neural network society",
                  "ai magazine",
                  "artif. intell."]
"""
 #########################
"""

journals = {"Wissensrepresentation und Schlussfolgern": ["autonomous agents and multi-agent systems",
                                                         "journal of automated reasoning"],
            "Unsicherheit und Schlussfolgern": [],
            "Maschinelles Lernen": ["machine learning",
                                    "neural networks : the official journal of the international neural network society",
                                    "ai magazine",
                                    "artif. intell.",
                                    "ieee transactions on neural networks and learning systems",
                                    "ieee transactions on neural networks",
                                    "journal of machine learning research",
                                    "fourth ieee international conference on data mining (icdm'04)",
                                    "fifth ieee international conference on data mining (icdm'05)",
                                    "sixth international conference on data mining (icdm'06)",
                                    "seventh ieee international conference on data mining (icdm 2007)",
                                    "2008 eighth ieee international conference on data mining",
                                    "2009 ninth ieee international conference on data mining",
                                    "2010 ieee international conference on data mining",
                                    "2011 ieee 11th international conference on data mining",
                                    "2012 ieee 12th international conference on data mining",
                                    "2013 ieee 13th international conference on data mining",
                                    "2014 ieee international conference on data mining",
                                    "2015 ieee international conference on data mining",
                                    "2016 ieee 16th international conference on data mining (icdm)",
                                    "2017 ieee international conference on data mining (icdm)",
                                    "ieee transactions on knowledge and data engineering",
                                    "neural networks",
                                    "data mining and knowledge discovery",
                                    "2004 ieee international joint conference on neural networks (ieee cat. no.04ch37541)",
                                    "2007 international joint conference on neural networks",
                                    "2008 ieee international joint conference on neural networks (ieee world congress on computational intelligence)",
                                    "2009 international joint conference on neural networks",
                                    "the 2010 international joint conference on neural networks (ijcnn)",
                                    "the 2011 international joint conference on neural networks",
                                    "the 2012 international joint conference on neural networks (ijcnn)",
                                    "the 2013 international joint conference on neural networks (ijcnn)",
                                    "2014 international joint conference on neural networks (ijcnn)",
                                    "2015 international joint conference on neural networks (ijcnn)",
                                    "2016 international joint conference on neural networks (ijcnn)",
                                    "2017 international joint conference on neural networks (ijcnn)",
                                    "neural processing letters",
                                    "information retrieval",
                                    "information retrieval journal"],
            "Wahrnemung und Sehen": ["2005 ieee computer society conference on computer vision and pattern recognition (cvpr'05)",
                                     "2006 ieee computer society conference on computer vision and pattern recognition (cvpr'06)",
                                     "2007 ieee conference on computer vision and pattern recognition",
                                     "2008 ieee conference on computer vision and pattern recognition",
                                     "2009 ieee conference on computer vision and pattern recognition",
                                     "2010 ieee computer society conference on computer vision and pattern recognition",
                                     "cvpr 2011",
                                     "2012 ieee conference on computer vision and pattern recognition",
                                     "2013 ieee conference on computer vision and pattern recognition",
                                     "2014 ieee conference on computer vision and pattern recognition",
                                     "2015 ieee conference on computer vision and pattern recognition (cvpr)",
                                     "2016 ieee conference on computer vision and pattern recognition (cvpr)",
                                     "2017 ieee conference on computer vision and pattern recognition (cvpr)",
                                     "2015 ieee international conference on computer vision (iccv)",
                                     "2017 ieee international conference on computer vision (iccv)",
                                     "ieee transactions on pattern analysis and machine intelligence",
                                     "international journal of computer vision",
                                     "computer vision and image understanding"
                                     ],
            "Robotik": ["the international journal of robotics research",
                        "ieee transactions on robotics",
                        "2004 ieee/rsj international conference on intelligent robots and systems (iros) (ieee cat. no.04ch37566)",
                        "2005 ieee/rsj international conference on intelligent robots and systems",
                        "2006 ieee/rsj international conference on intelligent robots and systems",
                        "2007 ieee/rsj international conference on intelligent robots and systems",
                        "2008 ieee/rsj international conference on intelligent robots and systems",
                        "2009 ieee/rsj international conference on intelligent robots and systems",
                        "2010 ieee/rsj international conference on intelligent robots and systems",
                        "2011 ieee/rsj international conference on intelligent robots and systems",
                        "2012 ieee/rsj international conference on intelligent robots and systems",
                        "2014 ieee/rsj international conference on intelligent robots and systems",
                        "2013 ieee/rsj international conference on intelligent robots and systems",
                        "2015 ieee/rsj international conference on intelligent robots and systems (iros)",
                        "2016 ieee/rsj international conference on intelligent robots and systems (iros)",
                        "2017 ieee/rsj international conference on intelligent robots and systems (iros)",
                        "robotics and autonomous systems"
                        ]}

 #################################################

venues = {"Wissensrepresentation und Schlussfolgern": ["autonomous agents and multi-agent systems",
                                                       "icaps",
                                                       "journal of automated reasoning",
                                                       "jelia",  # European Conf. on Logics and Artificial Intelligence
                                                       "tableaux",
                                                       "iccbr",
                                                       "tark",
                                                       "iclp"],
          "Unsicherheit und Schlussfolgern": ["uai"],
          "Maschinelles Lernen": ["nips",
                                  "icml",
                                  "aaai",
                                  "ijcai",
                                  "ai magazine",
                                  "machine learning",
                                  "ieee transactions on neural networks and learning systems",
                                  "ieee transactions on neural networks",
                                  "kdd",  # ???
                                  "sigir",
                                  "journal of machine learning research",
                                  "fourth ieee international conference on data mining (icdm'04)",
                                  "fifth ieee international conference on data mining (icdm'05)",
                                  "sixth international conference on data mining (icdm'06)",
                                  "seventh ieee international conference on data mining (icdm 2007)",
                                  "2008 eighth ieee international conference on data mining",
                                  "2009 ninth ieee international conference on data mining",
                                  "2010 ieee international conference on data mining",
                                  "2011 ieee 11th international conference on data mining",
                                  "2012 ieee 12th international conference on data mining",
                                  "2013 ieee 13th international conference on data mining",
                                  "2014 ieee international conference on data mining",
                                  "2015 ieee international conference on data mining",
                                  "2016 ieee 16th international conference on data mining (icdm)",
                                  "2017 ieee international conference on data mining (icdm)",
                                  "ieee transactions on knowledge and data engineering",
                                  "neural networks",
                                  "cikm",
                                  "aistats",
                                  "data mining and knowledge discovery",
                                  "sdm",
                                  "ecml/pkdd",
                                  "ecir",
                                  "pakdd",
                                  "recsys",
                                  "2004 ieee international joint conference on neural networks (ieee cat. no.04ch37541)",
                                  "2007 international joint conference on neural networks",
                                  "2008 ieee international joint conference on neural networks (ieee world congress on computational intelligence)",
                                  "2009 international joint conference on neural networks",
                                  "the 2010 international joint conference on neural networks (ijcnn)",
                                  "the 2011 international joint conference on neural networks",
                                  "the 2012 international joint conference on neural networks (ijcnn)",
                                  "the 2013 international joint conference on neural networks (ijcnn)",
                                  "2014 international joint conference on neural networks (ijcnn)",
                                  "2015 international joint conference on neural networks (ijcnn)",
                                  "2016 international joint conference on neural networks (ijcnn)",
                                  "2017 international joint conference on neural networks (ijcnn)",
                                  "neural processing letters",
                                  "information retrieval",
                                  "information retrieval journal",
                                  "icann",
                                  "ilp"],
          "Wahrnemung und Sehen": ["2005 ieee computer society conference on computer vision and pattern recognition (cvpr'05)",
                                   "2006 ieee computer society conference on computer vision and pattern recognition (cvpr'06)",
                                   "2007 ieee conference on computer vision and pattern recognition",
                                   "2008 ieee conference on computer vision and pattern recognition",
                                   "2009 ieee conference on computer vision and pattern recognition",
                                   "2010 ieee computer society conference on computer vision and pattern recognition",
                                   "cvpr 2011",
                                   "2012 ieee conference on computer vision and pattern recognition",
                                   "2013 ieee conference on computer vision and pattern recognition",
                                   "2014 ieee conference on computer vision and pattern recognition",
                                   "2015 ieee conference on computer vision and pattern recognition (cvpr)",
                                   "2016 ieee conference on computer vision and pattern recognition (cvpr)",
                                   "2017 ieee conference on computer vision and pattern recognition (cvpr)",
                                   "cvpr",
                                   "iccv",
                                   "2015 ieee international conference on computer vision (iccv)",
                                   "2017 ieee international conference on computer vision (iccv)",
                                   "ieee transactions on pattern analysis and machine intelligence",
                                   "eccv",
                                   "international journal of computer vision",
                                   "computer vision and image understanding"],
          "Verstehen und Generierung von natürlicher Sprache": ["acl",
                                                                "naacl",
                                                                "emnlp",
                                                                "coling",
                                                                "eacl",
                                                                "conll",
                                                                "acl/ijcnlp",
                                                                "ijcnlp",
                                                                "robotics: science and systems"
                                                                ],
          "Robotik": ["ieee transactions on robotics",
                      "2004 ieee/rsj international conference on intelligent robots and systems (iros) (ieee cat. no.04ch37566)",
                      "2005 ieee/rsj international conference on intelligent robots and systems",
                      "2006 ieee/rsj international conference on intelligent robots and systems",
                      "2007 ieee/rsj international conference on intelligent robots and systems",
                      "2008 ieee/rsj international conference on intelligent robots and systems",
                      "2009 ieee/rsj international conference on intelligent robots and systems",
                      "2010 ieee/rsj international conference on intelligent robots and systems",
                      "2011 ieee/rsj international conference on intelligent robots and systems",
                      "2012 ieee/rsj international conference on intelligent robots and systems",
                      "2014 ieee/rsj international conference on intelligent robots and systems",
                      "2013 ieee/rsj international conference on intelligent robots and systems",
                      "2015 ieee/rsj international conference on intelligent robots and systems (iros)",
                      "2016 ieee/rsj international conference on intelligent robots and systems (iros)",
                      "2017 ieee/rsj international conference on intelligent robots and systems (iros)",
                      "robotics and autonomous systems"
                      ]}

if False:
    lookup = "ewrl"
    lookup = lookup.lower()

    print("Journals:")
    print("=========")
    for jn in journal_dict:
        if lookup in jn:
            print(f"{jn}: {journal_dict[jn]}")
    print()
    print("Venues:")
    print("=======")
    for vn in venue_dict:
        if lookup in vn:
            print(f"{vn}: {venue_dict[vn]}")
    exit()

if False:
    total_v = 0
    total_j = 0
    for ven in check_venues:

        if ven in venue_dict:
            count = venue_dict[ven]
        else:
            count = 0
        print(f"{ven}: {count}")

        total_v += count
    print("----------")
    for journal in check_journals:
        if journal in journal_dict:
            count = journal_dict[journal]
        else:
            count = 0
        print(f"{journal}: {count}")

        total_j += count

    print()
    print(f"Total Venue: {total_v}")
    print(f"Total Journal: {total_j}")
    exit()

if True:
    total_v = 0
    total_j = 0

    print("Journals")
    print("///////////")
    for region in journals:
        print(region)
        print("===============")
        jr = journals[region]
        for mag in jr:
            total_j += journal_dict[mag]
            print(f"{mag}: {journal_dict[mag]}")
        print()

    print()
    print()
    print("Venues")
    print("///////////")
    for region in venues:
        print(region)
        print("===============")
        jr = venues[region]
        for mag in jr:
            total_v += venue_dict[mag]
            print(f"{mag}: {venue_dict[mag]}")
        print()

    print(f"Total Journals: {total_j}")
    print(f"Total Venues: {total_v}")
