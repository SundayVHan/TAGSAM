import data
import os
import json
import torch
import pandas as pd


def load_data(dataset, seed=0):
    if dataset == 'cora':
        pass
        # from data.data_utils.load_cora import get_raw_text_cora as get_raw_text
        # num_classes = 7
        # # class_map = 'Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory'
        # classes = ['Case Based', 'Genetic Algorithms', 'Neural Networks',
        #            'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning', 'Theory']
        # c_descs = [
        #     ' which refers to research papers focusing on case-based reasoning (CBR) in the field of artificial intelligence. Case-based reasoning is a problem-solving approach that utilizes specific knowledge of previously encountered, concrete problem situations (cases). In this method, a new problem is solved by finding similar past cases and reusing them in the new situation. The approach relies on the idea of learning from past experiences to solve new problems, which makes it relevant in many applications including medical diagnosis, legal decision-making, and others. Thus, the ""Case Based"" category would include papers that primarily focus on this particular methodology and its various aspects.',
        #     ' which would include research papers related to genetic algorithms (GAs). Genetic algorithms are a type of optimization and search algorithms inspired by the process of natural selection and genetics. These algorithms generate solutions to optimization problems using techniques inspired by natural evolution, such as inheritance, mutation, selection, and crossover. In practice, genetic algorithms can be used to find solutions to complex problems that are difficult to solve with traditional methods, particularly in domains where the search space is large, complex, or poorly understood. This category would cover various aspects of genetic algorithms, including their design, analysis, implementation, theoretical background, and diverse applications.',
        #     " which refers to research papers revolving around the concept of artificial neural networks (ANNs). Neural networks are a subset of machine learning algorithms modelled after the human brain, designed to ""learn"" from observational data. They are the foundation of deep learning technologies and can process complex data inputs, find patterns, and make decisions. The network consists of interconnected layers of nodes, or ""neurons"", and each connection is assigned a weight that shapes the data and helps produce a meaningful output. Topics covered under this category could range from the architecture and function of different neural network models, advancements in training techniques, to their application in a multitude of fields such as image and speech recognition, natural language processing, and medical diagnosis.",
        #     " which pertains to research papers that focus on probabilistic methods and models in machine learning and artificial intelligence. Probabilistic methods use the mathematics of probability to make predictions and decisions. They provide a framework to handle and quantify the uncertainty and incomplete information, which is a common scenario in real-world problems. This category could include topics like Bayesian networks, Gaussian processes, Markov decision processes, and statistical techniques for prediction and inference. These methods have applications in various areas such as computer vision, natural language processing, robotics, and data analysis, among others, due to their ability to model complex, uncertain systems and make probabilistic predictions.",
        #     " which refers to research papers focusing on the area of machine learning known as reinforcement learning (RL). Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve a goal. The agent learns from the consequences of its actions, rather than from being explicitly taught, and adjusts its behavior based on the positive or negative feedback it receives, known as rewards or penalties. This category would include research exploring various RL algorithms, methodologies, theoretical underpinnings, performance enhancements, and practical applications. This field is particularly relevant in areas where decision making is crucial, such as game playing, robotics, resource management, and autonomous driving.",
        #     " which pertains to research papers that concentrate on the domain of rule-based learning, also known as rule-based machine learning. Rule learning is a method in machine learning that involves the generation of a set of rules to predict the output in a decision-making system based on the patterns discovered from the data. These rules are often in an ""if-then"" format, making them interpretable and transparent. This category would encompass research involving various rule learning algorithms, their enhancements, theoretical foundations, and applications. Rule learning methods are particularly beneficial in domains where interpretability and understanding of the learned knowledge is important, such as in medical diagnosis, credit risk prediction, and more.",
        #     ' which likely refers to research papers that delve into the theoretical aspects of machine learning and artificial intelligence. This includes a broad array of topics such as theoretical foundations of various machine learning algorithms, performance analysis, studies on learning theory, statistical learning, information theory, and optimization methods. Additionally, it could encompass the development of new theoretical frameworks, investigations into the essence of intelligence, the potential for artificial general intelligence, as well as the ethical implications surrounding AI. Essentially, the ""Theory"" category encapsulates papers that primarily focus on theoretical concepts and discussions, contrasting with more application-oriented research which centers on specific techniques and their practical implementation.']
    # elif dataset == 'pubmed':
    #     from data.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
    #     num_classes = 3
    #     # class_map = 'Experimental induced diabetes, Type 1 diabetes, Type 2 diabetes'
    #     # class_map = 'Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2'
    #     classes = ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type1', 'Diabetes Mellitus Type2']
    #     c_descs = [
    #         ' which is a category of scientific literature found on PubMed that encompasses research related to experimental studies on diabetes mellitus. This category includes studies conducted in laboratory settings, often using animal models or cell cultures, to investigate various aspects of diabetes, such as its pathophysiology, treatment strategies, and potential interventions. Researchers in this field aim to better understand the underlying mechanisms of diabetes and develop experimental approaches to prevent or manage the disease. Experimental studies in this category may explore topics like insulin resistance, beta cell function, glucose metabolism, and the development of novel therapies for diabetes.',
    #         ' which focuses on scientific research related specifically to Type 1 diabetes mellitus. This category encompasses a wide range of studies, including clinical trials, epidemiological investigations, and basic research, all centered on understanding, diagnosing, managing, and potentially curing Type 1 diabetes. Researchers in this field explore areas such as the autoimmune processes underlying the disease, insulin therapy, glucose monitoring, pancreatic islet transplantation, and novel treatments aimed at improving the lives of individuals with Type 1 diabetes. It serves as a valuable resource for healthcare professionals, scientists, and policymakers interested in advancements related to Type 1 diabetes management and research.',
    #         ' which focuses on research related to Type 2 diabetes (T2D), and it can be differentiated from Diabetes Mellitus Type 1 (T1D) in the following ways: Etiology (Cause): Type 2 Diabetes (T2D): T2D is primarily characterized by insulin resistance, where the body\'s cells do not respond effectively to insulin, and relative insulin deficiency that develops over time. It is not primarily an autoimmune condition.']
    elif dataset == 'arxiv':
        from data.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
        classes = [
            "cs.NA(Numerical Analysis)",
            "cs.MM(Multimedia)",
            "cs.LO(Logic in Computer Science)",
            "cs.CY(Computers and Society)",
            "cs.CR(Cryptography and Security)",
            "cs.DC(Distributed, Parallel, and Cluster Computing)",
            "cs.HC(Human-Computer Interaction)",
            "cs.CE(Computational Engineering, Finance, and Science)",
            "cs.NI(Networking and Internet Architecture)",
            "cs.CC(Computational Complexity)",
            "cs.AI(Artificial Intelligence)",
            "cs.MA(Multiagent Systems)",
            "cs.GL(General Literature)",
            "cs.NE(Neural and Evolutionary Computing)",
            "cs.SC(Symbolic Computation)",
            "cs.AR(Hardware Architecture)",
            "cs.CV(Computer Vision and Pattern Recognition)",
            "cs.GR(Graphics)",
            "cs.ET(Emerging Technologies)",
            "cs.SY(Systems and Control)",
            "cs.CG(Computational Geometry)",
            "cs.OH(Other Computer Science)",
            "cs.PL(Programming Languages)",
            "cs.SE(Software Engineering)",
            "cs.LG(Machine Learning)",
            "cs.SD(Sound)",
            "cs.SI(Social and Information Networks)",
            "cs.RO(Robotics)",
            "cs.IT(Information Theory)",
            "cs.PF(Performance)",
            "cs.CL(Computational Complexity)",
            "cs.IR(Information Retrieval)",
            "cs.MS(Mathematical Software)",
            "cs.FL(Formal Languages and Automata Theory)",
            "cs.DS(Data Structures and Algorithms)",
            "cs.OS(Operating Systems)",
            "cs.GT(Computer Science and Game Theory)",
            "cs.DB(Databases)",
            "cs.DL(Digital Libraries)",
            "cs.DM(Discrete Mathematics)"
        ]
        c_descs = [
            "Numerical Analysis: Study of algorithms for approximating mathematical problems.",
            "Multimedia: Research involving multimedia technologies like image, audio, and video processing.",
            "Logic in Computer Science: Study of logic systems and their applications in computing, such as automated reasoning.",
            "Computers and Society: Exploration of the impact of computing technology on society and culture, including ethics and privacy issues.",
            "Cryptography and Security: Study of techniques for securing information, including encryption and network security.",
            "Distributed, Parallel, and Cluster Computing: Design and implementation of distributed systems, parallel computing, and cluster computing.",
            "Human-Computer Interaction: Study of interaction between humans and computers, aiming to improve user experience and usability.",
            "Computational Engineering, Finance, and Science: Application of computational techniques to solve problems in engineering, finance, and science.",
            "Networking and Internet Architecture: Research on network protocols, architectures, and the development of internet technologies.",
            "Computational Complexity: Study of the complexity of computational problems, including time and space complexity.",
            "Artificial Intelligence: Research in AI technologies such as machine learning, natural language processing, and computer vision.",
            "Multiagent Systems: Study of systems composed of multiple autonomous entities and their applications.",
            "General Literature: Comprehensive literature and reviews in the field of computer science.",
            "Neural and Evolutionary Computing: Application of neural networks and evolutionary algorithms in computation.",
            "Symbolic Computation: Study of symbolic mathematical computation and computer algebra systems.",
            "Hardware Architecture: Research on the structure of computer hardware and its interaction with software.",
            "Computer Vision and Pattern Recognition: Research and application of computer vision and pattern recognition technologies.",
            "Graphics: Study of computer graphics, including image generation, modeling, and rendering techniques.",
            "Emerging Technologies: Focus on new and emerging computing technologies and their potential applications.",
            "Systems and Control: Research and application of systems engineering and control theory.",
            "Computational Geometry: Study of algorithms and computational methods for geometric problems.",
            "Other Computer Science: Topics in computer science not covered by other categories.",
            "Programming Languages: Study of the design, implementation, and analysis of programming languages.",
            "Software Engineering: Research on software development processes, tools, and methodologies.",
            "Machine Learning: Study of machine learning algorithms and their applications in various fields.",
            "Sound: Research involving sound processing and audio technologies.",
            "Social and Information Networks: Study of the structure and dynamics of social and information networks.",
            "Robotics: Research and application of robotics technologies, including automation and control.",
            "Information Theory: Study of the quantification, storage, and communication of information.",
            "Performance: Research on performance analysis and optimization of computing systems.",
            "Computational Complexity: Study of the complexity of computational problems, including time and space complexity.",
            "Information Retrieval: Research on information retrieval techniques, including search engines and data mining.",
            "Mathematical Software: Study of software tools and systems for mathematical computation.",
            "Formal Languages and Automata Theory: Study of formal languages and automata theory, including grammar and parsing techniques.",
            "Data Structures and Algorithms: Design and analysis of data structures and algorithms.",
            "Operating Systems: Research on the design, implementation, and management of operating systems.",
            "Computer Science and Game Theory: Study of game theory applications in computer science.",
            "Databases: Research on the design, implementation, and optimization of database systems.",
            "Digital Libraries: Study of the construction and management of digital libraries, including information storage and retrieval.",
            "Discrete Mathematics: Study of discrete mathematical methods and applications in computer science."
        ]
        # for i in range(len(c_descs)):
        #     c_descs[i] = classes[i] + c_descs[i]

    elif dataset == 'products':
        from data.data_utils.load_products import get_raw_text_products as get_raw_text
        num_classes = 47
        classes = [
            'Home & Kitchen',
            'Health & Personal Care',
            'Beauty',
            'Sports & Outdoors',
            'Books',
            'Patio, Lawn & Garden',
            'Toys & Games',
            'CDs & Vinyl',
            'Cell Phones & Accessories',
            'Grocery & Gourmet Food',
            'Arts, Crafts & Sewing',
            'Clothing, Shoes & Jewelry',
            'Electronics',
            'Movies & TV',
            'Software',
            'Video Games',
            'Automotive',
            'Pet Supplies',
            'Office Products',
            'Industrial & Scientific',
            'Musical Instruments',
            'Tools & Home Improvement',
            'Magazine Subscriptions',
            'Baby Products',
            "nan",
            'Appliances',
            'Kitchen & Dining',
            'Collectibles & Fine Art',
            'All Beauty',
            'Luxury Beauty',
            'Amazon Fashion',
            'Computers',
            'All Electronics',
            'Purchase Circles',
            'MP3 Players & Accessories',
            'Gift Cards',
            'Office & School Supplies',
            'Home Improvement',
            'Camera & Photo',
            'GPS & Navigation',
            'Digital Music',
            'Car Electronics',
            'Baby',
            'Kindle Store',
            'Buy a Kindle',
            'Furniture & Decor',  # Converted HTML entity to plain text
            'nan'
        ]
        c_descs = [
            ". Home & Kitchen includes a wide range of products designed to enhance the living space, from furniture and appliances to cooking gadgets and home decor.",
            ". Health & Personal Care offers products focused on wellness, hygiene, and personal grooming, including skincare, supplements, and medical supplies.",
            ". Beauty encompasses cosmetics, skincare, and haircare products, helping individuals express their style and maintain their appearance.",
            ". Sports & Outdoors features equipment and apparel for various sports and outdoor activities, promoting fitness and adventure.",
            ". Books provide a diverse selection of literature, from fiction and non-fiction to educational and reference materials.",
            ". Patio, Lawn & Garden includes tools and decor for outdoor spaces, enhancing the aesthetic and functionality of gardens and patios.",
            ". Toys & Games offer entertainment and educational value for children and adults, ranging from board games to electronic toys.",
            ". CDs & Vinyl cater to music enthusiasts with a collection of physical music formats, preserving the classic listening experience.",
            ". Cell Phones & Accessories cover the latest mobile devices and their peripherals, ensuring connectivity and functionality on the go.",
            ". Grocery & Gourmet Food provides a variety of food products, from everyday groceries to specialty gourmet items.",
            ". Arts, Crafts & Sewing supply materials and tools for creative projects, supporting hobbies like painting, knitting, and DIY crafts.",
            ". Clothing, Shoes & Jewelry feature fashion items for all occasions, helping individuals express their personal style.",
            ". Electronics includes a wide array of gadgets and devices, from computers and cameras to home entertainment systems.",
            ". Movies & TV offer a collection of films and television series, available in various formats for home viewing.",
            ". Software provides applications and programs for a range of tasks, from productivity and creativity to entertainment and education.",
            ". Video Games encompass digital games for various platforms, offering interactive entertainment and immersive experiences.",
            ". Automotive includes car parts, accessories, and tools, supporting vehicle maintenance and enhancement.",
            ". Pet Supplies offer products for pet care and comfort, including food, toys, and grooming essentials.",
            ". Office Products provide supplies and equipment for workplace efficiency and organization, from stationery to furniture.",
            ". Industrial & Scientific includes specialized equipment and tools for professional and scientific applications.",
            ". Musical Instruments offer a range of instruments and accessories for musicians, from beginners to professionals.",
            ". Tools & Home Improvement provide equipment and materials for DIY projects and home renovations.",
            ". Magazine Subscriptions offer regular deliveries of periodicals on various topics, from fashion to technology.",
            ". Baby Products include essentials for infant care and development, from diapers to toys.",
            ". nan represents a category with missing or undefined information.",
            ". Appliances feature household devices that assist with daily tasks, such as refrigerators, washers, and microwaves.",
            ". Kitchen & Dining includes cookware, utensils, and tableware, enhancing the cooking and dining experience.",
            ". Collectibles & Fine Art offer unique items for collectors and art enthusiasts, from paintings to rare memorabilia.",
            ". All Beauty encompasses a wide range of beauty products, catering to skincare, makeup, and haircare needs.",
            ". Luxury Beauty provides high-end beauty products with premium ingredients and packaging.",
            ". Amazon Fashion features a curated collection of clothing and accessories for various styles and occasions.",
            ". Computers include desktop and laptop systems, along with peripherals and components for tech enthusiasts.",
            ". All Electronics covers a broad spectrum of electronic devices and accessories, supporting modern lifestyles.",
            ". Purchase Circles offer group buying opportunities for discounted products and shared interests.",
            ". MP3 Players & Accessories provide portable audio solutions for music lovers on the go.",
            ". Gift Cards offer flexible gifting options, allowing recipients to choose their preferred products.",
            ". Office & School Supplies feature essential items for educational and professional environments.",
            ". Home Improvement includes tools and materials for enhancing and maintaining homes.",
            ". Camera & Photo offer photography equipment and accessories for capturing memories.",
            ". GPS & Navigation provide devices and software for accurate location tracking and route planning.",
            ". Digital Music offers a vast collection of music available for download and streaming.",
            ". Car Electronics include audio systems, GPS devices, and other in-car technology.",
            ". Baby features products specifically designed for infants and toddlers, ensuring safety and comfort.",
            ". Kindle Store provides digital books and reading devices for convenient access to literature.",
            ". Buy a Kindle offers options for purchasing Amazon's e-reader devices.",
            ". Furniture & Decor includes items for furnishing and decorating living spaces, enhancing comfort and style.",
            ". #508510 represents a unique or special category, potentially signifying a placeholder or specific item."
        ]
    # elif dataset == 'arxiv_2023':
    #     from data.data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
    #     num_classes = 40
    #     classes = []
    #     c_descs = []
    #
    # elif dataset == 'citeseer':
    #     from data.data_utils.load_citeseer import get_raw_text_citeseer as get_raw_text
    #     classes = ['Agents', 'Machine Learning', 'Information Retrieval', 'Database', 'Human Computer Interaction',
    #                'Artificial Intelligence']
    #     c_descs = [
    #         ". Specifically, agents are autonomous entities that perceive their environment through sensors and act upon it using actuators. They are designed to achieve specific goals or tasks.",
    #         ". Specifically, ML research investigates how to create systems that can automatically improve their performance on tasks by identifying patterns and insights from vast amounts of data. Researchers in Machine Learning explore diverse techniques such as supervised learning, unsupervised learning, reinforcement learning, and deep learning to build systems that can predict outcomes, classify data, and make intelligent decisions.",
    #         ". Specifically, IR research focuses on the study of information retrieval systems, which are designed to help users find relevant information in large collections of data. Researchers in Information Retrieval explore techniques such as indexing, querying, and ranking to build systems that can efficiently retrieve information based on user queries.",
    #         ". Specifically, DB research investigates how to design, build, and manage databases, which are organized collections of data that can be accessed, managed, and updated. Researchers in Database Systems explore techniques such as data modeling, query languages, and transaction processing to build systems that can store, retrieve, and manipulate data.",
    #         ". Specifically, HCI research focuses on the study of human-computer interaction, which explores how people interact with computers and other digital technologies. Researchers in Human-Computer Interaction investigate how to design user-friendly interfaces, improve usability, and enhance user experience to build systems that are intuitive, efficient, and effective.",
    #         ". Specifically, AI research investigates how to create intelligent systems that can perform tasks that typically require human intelligence, such as perception, reasoning, learning, and decision-making. Researchers in Artificial Intelligence explore diverse techniques such as knowledge representation, planning, and natural language processing to build systems that can solve complex problems, adapt to new environments, and interact with humans.",
    #     ]
    #     num_classes = 6
    elif dataset == 'wikics':
        from data.data_utils.load_wikics import get_raw_text_wikics as get_raw_text
        classes = ['Computational linguistics', 'Databases', 'Operating systems', 'Computer architecture',
                   'Computer security, Computer network security, Access control, Data security, Computational trust, Computer security exploits',
                   'Internet protocols', 'Computer file systems', 'Distributed computing architecture',
                   'Web technology, Web software, Web services',
                   'Programming language topics, Programming language theory, Programming language concepts, Programming language classification']
        c_descs = [
            ". Computational linguistics is an interdisciplinary field combining linguistics and computer science to analyze and model natural language. It involves developing algorithms and computational models to understand, generate, and manipulate human language. Applications include machine translation, speech recognition, sentiment analysis, and chatbot development. By leveraging statistical methods and artificial intelligence, computational linguistics aims to enhance human-computer interaction and improve the processing of linguistic data.",
            ". Databases are organized collections of data, designed to store, manage, and retrieve information efficiently. They enable structured querying and data manipulation through languages like SQL. Databases can be categorized into relational (e.g., MySQL, PostgreSQL) and non-relational (e.g., MongoDB, Cassandra) systems, each suited for different applications and data structures. They play a vital role in various domains, including business, research, and web applications, facilitating data-driven decision-making.",
            ". Operating systems (OS) are essential software that manage computer hardware and software resources, providing a user interface and facilitating interactions between applications and hardware. Key functions include process management, memory management, file system handling, and device control. Popular operating systems include Windows, macOS, and Linux. OSs enable multitasking, security, and resource allocation, playing a crucial role in the overall functionality and performance of computing devices.",
            ". Computer architecture is the design and organization of computer systems, encompassing the structure and functionality of hardware components. It includes the CPU, memory hierarchy, and input/output systems, focusing on how they interact to perform tasks efficiently. Key concepts involve instruction sets, parallelism, and microarchitecture. Understanding computer architecture is crucial for optimizing performance, enhancing energy efficiency, and developing new computing technologies, impacting both hardware design and software development.",
            ". Computer security encompasses measures to protect systems from threats, ensuring confidentiality, integrity, and availability of data. Computer network security focuses on safeguarding networks from unauthorized access and attacks. Access control regulates who can view or use resources, while data security protects sensitive information from breaches. Computational trust ensures reliability in transactions and interactions, and computer security exploits are vulnerabilities that attackers leverage to compromise systems. Together, these elements safeguard digital environments.",
            ". Internet protocols are standardized rules that govern data communication over the internet, ensuring devices can communicate effectively. Key examples include TCP (Transmission Control Protocol), which ensures reliable data transmission, and IP (Internet Protocol), which handles addressing and routing. Other protocols, like HTTP (for web traffic) and FTP (for file transfer), facilitate specific types of data exchange. Collectively, these protocols enable the seamless functioning of the internet and support diverse applications and services.",
            ". Computer file systems are crucial components of operating systems that manage how data is stored, organized, and accessed on storage devices. They arrange files into directories, facilitate operations like creation and deletion, and manage permissions and metadata. Various file systems exist, such as NTFS (Windows), ext4 (Linux), and HFS+ (macOS), each designed for specific performance, reliability, and compatibility needs across different platforms.",
            ". Distributed computing architecture involves a system of interconnected computers that collaboratively process data and tasks. It enables resource sharing and parallel processing across multiple machines, enhancing performance and scalability. Key components include clients, servers, and communication protocols that facilitate coordination and data exchange. Common examples are cloud computing and grid computing. This architecture is vital for handling large-scale applications, improving efficiency, and supporting fault tolerance in various domains, from scientific research to enterprise solutions.",
            ". Web technology encompasses tools and protocols that facilitate the creation and interaction of web applications and services. Web software refers to applications designed to run on web servers, such as content management systems and e-commerce platforms. Web services are standardized methods for enabling communication between different software systems over the internet, typically using protocols like HTTP and XML or JSON for data exchange. Together, they underpin the functionality and connectivity of the modern web.",
            ". Programming language topics encompass the study of languages used for software development, focusing on syntax, semantics, and implementation. Programming language theory investigates foundational concepts, including type systems, compilers, and language design. Programming language concepts cover key ideas like abstraction, encapsulation, and concurrency, shaping how languages are built and used. Programming language classification categorizes languages based on paradigms (e.g., procedural, functional, object-oriented), syntax, and application domains, aiding in understanding their strengths and weaknesses.", ]
    elif dataset == 'photo':
        from data.data_utils.load_photo import get_raw_text_photo as get_raw_text
        classes = [
            "Video Surveillance",
            "Accessories",
            "Binoculars & Scopes",
            "Video",
            "Lighting & Studio",
            "Bags & Cases",
            "Tripods & Monopods",
            "Flashes",
            "Digital Cameras",
            "Film Photography",
            "Lenses",
            "Underwater Photography"
        ]
        c_descs = [
            "which enhance security and monitoring with advanced video surveillance systems. From IP cameras and security camera systems to video recorders and monitoring software, these solutions offer peace of mind and visual protection.",
            "which enhance your photography experience with a vast array of accessories. Find memory cards, batteries, camera grips, remote controls, and more to ensure you're equipped for any shooting scenario.",
            "which bring distant subjects into focus with high-quality binoculars and scopes. These optical instruments are perfect for birdwatching, nature observation, stargazing, and outdoor adventures.",
            "which elevate your videography skills with professional video equipment. From high-quality camcorders and action cameras to gimbals, drones, and editing software, these tools empower you to create stunning visual content.",
            "which take your photography to the next level with professional lighting and studio equipment. Explore continuous lighting, strobes, softboxes, reflectors, and backdrops for achieving perfect illumination and creative effects.",
            "which protect your valuable photography equipment with durable and stylish bags and cases. Choose from backpacks, messenger bags, roller cases, and protective sleeves to keep your gear safe and organized.",
            "which achieve steady and blur-free shots with sturdy tripods and monopods. These essential tools provide stability and versatility, enabling you to capture sharp images and smooth video footage.",
            "which illuminate your subjects with precision and control using high-performance camera flashes. From compact on-camera flashes to powerful studio strobes, these lighting solutions enhance your photography in any environment.",
            "which upgrade your photography game with advanced digital cameras. Explore a diverse selection of DSLRs, mirrorless cameras, point-and-shoots, and camera bundles to capture precious moments with stunning clarity.",
            "which rediscover the art of traditional film photography. Explore a wide range of film cameras, film stocks, darkroom equipment, and accessories for capturing timeless images with a vintage aesthetic.",
            "which expand your creative possibilities with a diverse range of camera lenses. From wide-angle to telephoto, prime to zoom, these lenses offer versatility and precision for capturing breathtaking images.",
            "which dive into the captivating world of underwater photography. Discover specialized waterproof cameras, housings, lenses, and accessories designed to withstand the aquatic environment and capture stunning marine life."
        ]
    elif dataset == 'computer':
        from data.data_utils.load_computer import get_raw_text_computer as get_raw_text
        classes = [
            "Computer Accessories & Peripherals",
            "Tablet Accessories",
            "Laptop Accessories",
            "Computers & Tablets",
            "Computer Components",
            "Data Storage",
            "Networking Products",
            "Monitors",
            "Servers",
            "Tablet Replacement Parts"
        ]
        c_descs = [
            "which upgrade your computing setup with essential accessories and peripherals. Explore keyboards, mice, webcams, printers, scanners, and more to boost productivity, enhance ergonomics, and streamline your workflow.",
            "which enhance the functionality and protection of your tablet with a variety of accessories. Cases, covers, stylus pens, keyboards, stands, and screen protectors ensure a personalized and secure tablet experience.",
            "which enhance your laptop experience with a wide range of accessories. From protective cases and sleeves to external hard drives, wireless mice, and portable speakers, these accessories provide convenience, functionality, and personalization for your laptop.",
            "which explore a wide range of computers and tablets to suit your personal or professional requirements. From powerful desktop PCs and portable laptops to versatile 2-in-1 devices and sleek tablets, these cutting-edge machines offer performance, mobility, and functionality.",
            "which build or upgrade your custom computer with high-performance components. From processors and motherboards to graphics cards, RAM, and cooling solutions, these components offer power, speed, and customization for your unique computing needs.",
            "which keep your digital files safe and organized with reliable data storage solutions. Choose from external hard drives, solid-state drives, network-attached storage, and cloud storage options to securely store and backup your valuable data.",
            "which establish seamless connectivity with networking products designed for home or office use. Routers, modems, switches, access points, and networking cables ensure reliable internet access, file sharing, and secure network connections.",
            "which upgrade your visual experience with high-quality monitors suitable for various computing needs. From sleek and stylish displays for everyday use to specialized monitors for gaming, graphic design, or professional applications, these monitors deliver stunning visuals and optimal viewing angles.",
            "which power your business or organization with reliable and high-performance servers. From entry-level servers for small businesses to enterprise-grade solutions for data centers, these servers offer robust computing power, storage capacity, and advanced features.",
            "which extend the life of your tablet by replacing worn-out or damaged components. Find official replacement parts like batteries, screens, cameras, and other essential components to restore your tablet's functionality."
        ]
    elif dataset == 'history':
        from data.data_utils.load_history import get_raw_text_history as get_raw_text
        classes = [
            "World",
            "Americas",
            "Asia",
            "Military",
            "Europe",
            "Russia",
            "Africa",
            "Ancient Civilizations",
            "Middle East",
            "Historical Study & Educational Resources",
            "Australia & Oceania",
            "Arctic & Antarctica"
        ]
        c_descs = [
            "which explores global events and trends throughout history.",
            "which delves into the rich history of North, Central, and South America.",
            "which focuses on the diverse cultures and historical developments in Asia.",
            "which examines wars, conflicts, military strategy, and their impact on history.",
            "which covers the complex history of the European continent.",
            "which specifically studies the history of Russia, its empires, and its role in the world.",
            "which explores the vast and diverse histories of African nations and cultures.",
            "which investigates the origins, rise, and fall of early civilizations.",
            "which examines the history of the Middle East, including its cultures and pivotal events.",
            "which provides tools, guides, and resources for the study of history.",
            "which focuses on the history of Australia, New Zealand, and Pacific Island nations.",
            "which explores the history and significance of the polar regions."
        ]
    elif dataset == "instagram":
        from data.data_utils.load_instagram import get_raw_text_instagram as get_raw_text
        classes = ['Normal Users', 'Commercial Users']
        c_descs = [
            " who typically shares personal moments and engages with friends and family, focusing on social connections and self-expression through photos and stories. Their primary goal is to enjoy and explore content that reflects their interests and lifestyle.",
            " who leverages the platform to promote products or services, utilizing targeted advertising and engaging content to reach potential customers. Their focus is on brand growth and customer interaction, often employing analytics to refine strategies and enhance reach."]
        # for i in range(len(c_descs)):
        #     c_descs[i] = classes[i] + c_descs[i]
    # elif dataset == 'reddit':
    #     from data.data_utils.load_reddit import get_raw_text_reddit as get_raw_text
    #     classes = ['Normal Users', 'Popular Users']
    #     c_descs = ["", ""]
    else:
        exit(f'Error: Dataset {dataset} not supported')

    data, text = get_raw_text(use_text=True, seed=seed)

    return data, text, classes, c_descs

