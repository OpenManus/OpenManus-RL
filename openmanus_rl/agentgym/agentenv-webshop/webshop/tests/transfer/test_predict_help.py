import json
import pytest
import requests_mock

from transfer.predict_help import (
    convert_dict_to_actions,
    parse_item_page_amz, parse_results_amz,
    parse_item_page_ebay, parse_results_ebay,
    parse_item_page_ws, parse_results_ws,
    Page, WEBSHOP_URL, WEBSHOP_SESSION
)

@requests_mock.Mocker(kw="mock")
def test_parse_item_page_ws(**kwargs):
    # Read mock response data
    mock_file = open("tests/transfer/mocks/mock_parse_item_page_ws", "rb")
    mock_body = mock_file.read()
    mock_file.close()

    mock_desc_file = open("tests/transfer/mocks/mock_parse_item_page_ws_desc", "rb")
    mock_desc_body = mock_desc_file.read()
    mock_desc_file.close()

    mock_feat_file = open("tests/transfer/mocks/mock_parse_item_page_ws_feat", "rb")
    mock_feat_body = mock_feat_file.read()
    mock_feat_file.close()

    mock_asin = "B09P87V3LZ"
    mock_query = "red basketball shoes"
    mock_options = {}

    # Invoke function, check response
    query_str = '+'.join(mock_query.split())
    options_str = json.dumps(mock_options)
    url = (
        f"{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/"
        f"{mock_asin}/{query_str}/1/{options_str}"
    )
    url_desc = (
        f"{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/"
        f"{mock_asin}/{query_str}/1/Description/{options_str}"
    )
    url_feat = (
        f"{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/"
        f"{mock_asin}/{query_str}/1/Features/{options_str}"
    )
    print(f"Item Page URL: {url}")
    print(f"Item Description URL: {url_desc}")
    print(f"Item Features URL: {url_feat}")

    kwargs["mock"].get(url, content=mock_body)
    kwargs["mock"].get(url_desc, content=mock_desc_body)
    kwargs["mock"].get(url_feat, content=mock_feat_body)

    output = parse_item_page_ws(mock_asin, mock_query, 1, mock_options)
    expected = {
        'MainImage': 'https://m.media-amazon.com/images/I/51ltvkzGhGL.jpg',
        'Price': '100.0',
        'Rating': 'N.A.',
        'Title': 'PMUYBHF Womens Fashion Flat Shoes Comfortable Running Shoes ' 
            'Sneakers Tennis Athletic Shoe Casual Walking Shoes',
        'asin': mock_asin,
        'option_to_image': {
            '6.5': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%276.5%27%7D',
            '7.5': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%277.5%27%7D',
            '8': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%278%27%7D',
            '8.5': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%278.5%27%7D',
            '9': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27size%27:%20%279%27%7D',
            'black': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27color%27:%20%27black%27%7D',
            'purple': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27color%27:%20%27purple%27%7D',
            'red': 'http://3.83.245.205:3000/item_page/abc/B09P87V3LZ/%5B%27red%27%2C%20%27basketball%27%2C%20%27shoes%27%5D/1/%7B%27color%27:%20%27red%27%7D'
        },
        'options': {
            'color': ['black', 'purple', 'red'],
            'size': ['6.5', '7.5', '8', '8.5', '9']
        },
        'BulletPoints': 'Pure Running Shoe\nComfort Flat Sneakers\n[FEATURES]: Soles with unique non-slip pattern, it has great ' 
                'abrasion resistant and provide protection when you walking or running. (Pure Running Shoe Mesh Walking Shoes Fashion ' 
                'Sneakers Slip On Sneakers Wedge Platform Loafers Modern Walking Shoes Sock Sneakers Platform Loafers Shoes Non Slip ' 
                'Running Shoes Athletic Tennis Shoes Blade Type Sneakers Lace-up Sneaker) sole\n[WIDE ANKLE DESIGN]: Perfect accord with human body ' 
                'engineering, green, healthy concept design make the walking shoes wear more comfortable, wide width wlking shoes. (Low ' 
                'Top Walking Shoes Fashion Canvas Sneakers Slip On Shoes Casual Walking Shoes Hidden Wedge Sneaker Low Top Canvas ' 
                'Sneakers Lace-up Classic Casual Shoes Walking Tennis Shoes Lightweight Casual Sneakers Slip on Sock Sneakers Air ' 
                'Cushion Platform Loafers Slip-On Mule Sneaker )\n[CUSHION WITH ARCH SUPPORT]: Gives you a comfort for all day ' 
                'long. Wear these lightweight walking shoes, let every step of moving on a comfortable feeling. (Fashion Casual Shoes ' 
                'Athletic Workout Shoes Fitness Sneaker Athletic Running Shoes Air Cushion Sneakers Stylish Athletic Shoes Lace Up ' 
                'Canvas Shoes Slip on Walking Shoe Fashion Sneakers Low Top Classic Sneakers Comfort Fall Shoes Memory Foam Slip On ' 
                'Sneakers Air Cushion Sneakers Running Walking Shoes)\n[NON-SLIP SOLE]: Made from ultra soft and lightweight RUBBER ' 
                'material,with the function of shock absorbing and cushioning,offering the best durability and traction. (Wedge ' 
                'Sneakers Walking Tennis Shoes Slip On Running Shoes Lightweight Fashion Sneakers Fashion Travel Shoes Walking ' 
                'Running Shoes Non Slip Running Shoes Athletic Tennis Sneakers Sports Walking Shoes Platform Fashion Sneaker ' 
                'Memory Foam Tennis Sneakers Running Jogging Shoes Sock Sneakers Canvas Fashion Sneakers)\n[OCCASIONS]: Ultra lightweight design provides actual ' 
                'feelings of being barefooted and like walking on the feather, perfect for walking, hiking, bike riding, working, ' 
                'shopping, indoor, outdoor, casual, sports, travel, exercise, vacation, and etc. (Flat Fashion Sneakers Lightweight ' 
                'Walking Sneakers Platform Loafers Sport Running Shoes Casual Flat Loafers Slip-On Sneaker Casual Walking Shoes High Top ' 
                'Canvas Sneakers Lace up Sneakers Workout Walking Shoes Tennis Fitness Sneaker)\n' 
                '[Customers Are Our Priority]: We follow the principle of customer first, so if you encounter any problems after ' 
                'buying shoes, we will try our best to solve them for you. (Breathable Air Cushion Sneakers Walking Tennis Shoes Air ' 
                'Athletic Running Shoes Air Cushion Shoes Mesh Sneakers Fashion Tennis Shoes Jogging Walking Sneakers Breathable ' 
                'Casual Sneakers Fashion Walking Shoes Athletic Running Sneakers Walking Work Shoes Air Running Shoes Slip on ' 
                'Sneakers Mesh Walking Shoes)',
        'Description': 'Here Are The Things You Want To Knowa─=≡Σ(((つ̀ώ)つSTORE INTRODUCTION:>>>>Our store helps our customers improve their ' 
               'quality of life~As a distributor, we value quality and service.Focus on the high quality and durability of the ' 
               'product.Committed to creating a store that satisfies and reassures our customers.TIPS:>>>>1. Please allow minor errors ' 
               'in the data due to manual measurements.2. Due to the color settings of the display, the actual color may be slightly ' 
               'different from the online image.QUALITY PROMISE:>>>>Our goal is to continuously provide a range of quality products.We ' 
               'place a huge emphasis on the values of quality and reliability.We have always insisted on fulfilling this ' 
               'commitment.In short, we want our customers to have the same great product experience every time and be trusted to deliver ' 
               'on this commitment.Please give us a chance to serve you.OTHER:>>>>athletic sneaker laces athletic sneakers white ' 
               'athletic sneakers for women clearance leather Sneaker leather sneakers women leather sneakers for menleather sneaker laces ' 
               'leather sneaker platform basketball shoes basketball shoes for men basketball shoe laces basketball shoe grip basketball ' 
               'shoes for women fitness shoes for men fitness shoes women workout fitness shoes women fitness shoes women size 5 ' 
               'fitness shoes men workout fitness shoes for men high top sneakers for women walking shoes sneakers with arch support for women'
    }
    assert output == expected

@requests_mock.Mocker(kw="mock")
def test_parse_item_page_ebay(**kwargs):
    # Read mock response data
    mock_file = open("tests/transfer/mocks/mock_parse_item_page_ebay", "rb")
    mock_body = mock_file.read()
    mock_file.close()
    mock_asin = "403760625150"

    # Invoke function, check response
    kwargs["mock"].get(f"https://www.ebay.com/itm/{mock_asin}", content=mock_body)
    output = parse_item_page_ebay(mock_asin)
    expected = {
        'BulletPoints': 'Item specifics Condition:New without box: A brand-new, ' 
            'unused, and unworn item (including handmade items) that is ' 
            'not in ...  Read moreabout the conditionNew without box: A ' 
            'brand-new, unused, and unworn item (including handmade ' 
            'items) that is not in original packaging or may be missing ' 
            'original packaging materials (such as the original box or ' 
            'bag). The original tags may not be attached. For example, ' 
            'new shoes (with absolutely no signs of wear) that are no ' 
            'longer in their original box fall into this category.  See ' 
            'all condition definitionsopens in a new window or tab  ' 
            'Closure:Lace Up US Shoe Size:10 Occasion:Activewear, Casual ' 
            'Silhouette:Puma Fabric Type:Mesh Vintage:No Cushioning ' 
            'Level:Moderate Department:Men Style:Sneaker Outsole ' 
            'Material:Rubber Features:Breathable, Comfort, Cushioned, ' 
            'Performance Season:Fall, Spring, Summer, Winter ' 
            'Idset_Mpn:193990-21 Shoe Shaft Style:Low Top Style ' 
            'Code:193990-16 Pattern:Solid Character:J. Cole Lining ' 
            'Material:Synthetic Color:Red Brand:PUMA Type:Athletic ' 
            'Customized:No Model:RS-Dreamer Theme:Sports Shoe ' 
            'Width:Standard Upper Material:Textile Insole ' 
            'Material:Synthetic Performance/Activity:Basketball Product ' 
            'Line:Puma Dreamer',
        'Description': 'N/A',
        'MainImage': 'https://i.ebayimg.com/images/g/4ggAAOSwpk1ioTWz/s-l500.jpg',
        'Price': 'N/A',
        'Rating': None,
        'Title': "Puma RS-Dreamer J. Cole Basketball Shoes Red 193990-16 Men's Size 10.0",
        'asin': '403760625150',
        'option_to_image': {},
        'options': {},
    }
    assert output == expected

@requests_mock.Mocker(kw="mock")
def test_parse_item_page_amz(**kwargs):
    # Read mock response data
    mock_file = open("tests/transfer/mocks/mock_parse_item_page_amz", "rb")
    mock_body = mock_file.read()
    mock_file.close()
    mock_asin = "B073WRF565"

    # Invoke function, check response
    kwargs["mock"].get(f"https://www.amazon.com/dp/{mock_asin}", content=mock_body)
    output = parse_item_page_amz(mock_asin)
    expected = {
        'asin': 'B073WRF565',
        'Title': 'Amazon Basics Foldable 14" Black Metal Platform Bed Frame with Tool-Free Assembly No Box Spring Needed - Full',
        'Price': 'N/A',
        'Rating': '4.8 out of 5 stars',
        'BulletPoints': ' \n About this item    ' 
            'Product dimensions: 75" L x 54" W x 14" H | Weight: 41.4 pounds    ' 
            'Designed for sleepers up to 250 pounds    Full size platform bed frame offers a quiet, noise-free, ' 
            'supportive foundation for a mattress. No box spring needed    Folding mechanism makes the frame easy ' 
            'to store and move in tight spaces    Provides extra under-the-bed storage space with a vertical clea' 
            'rance of about 13 inches    \n › See more product details ',
        'Description': 'Amazon Basics Foldable, 14" Black Metal Platform Bed Frame with Tool-Free Assembly, No Box Spring Needed - Full   Amazon Basics',
        'MainImage': 'https://images-na.ssl-images-amazon.com/images/I/41WIGwt-asL.__AC_SY300_SX300_QL70_FMwebp_.jpg',
        'options': {'size': ['Twin', 'Full', 'Queen', 'King'],
        'style': ['14-Inch', '18-Inch']},
        'option_to_image': {}
    }
    assert output == expected

@requests_mock.Mocker(kw="mock")
def test_parse_results_ebay(**kwargs):
    # Read mock response data
    mock_file = open("tests/transfer/mocks/mock_parse_results_ebay", "rb")
    mock_body = mock_file.read()
    mock_file.close()
    mock_query = "red basketball shoes"
    
    # Invoke function, check response
    query = mock_query.replace(" ", "+")
    kwargs["mock"].get(f'https://www.ebay.com/sch/i.html?_nkw={query}&_pgn=1', content=mock_body)
    output = parse_results_ebay(mock_query, 1)
    expected = [{
		'Price': ['100.00', '150.00'],
		'Title': "Reebok Answer IV Men's Basketball Shoes",
		'asin': '175065123030'
	}, {
		'Price': '$119.90',
		'Title': "Air Jordan Stay Loyal Shoes Black Red White DB2884-001 Men's Multi "
		'Size NEW',
		'asin': '265672133690'
	}, {
		'Price': '$100.00',
		'Title': "Fila Men's Stackhouse Spaghetti Basketball Shoes Black Red White "
		'1BM01788-113',
		'asin': '175282509234'
	}, {
		'Price': ['61.99',
			'85.99'
		],
		'Title': 'Puma Disc Rebirth 19481203 Mens Black Red Synthetic Athletic '
		'Basketball Shoes',
		'asin': '313944854658'
	}, {
		'Price': '$0.01',
		'Title': "Puma RS-Dreamer J. Cole Basketball Shoes Red 193990-16 Men's Size "
		'10.0',
		'asin': '403760625150'
	}, {
		'Price': '$45.00',
		'Title': 'Nike Mens 9.5 PG 5  Maroon Red White Basketball Shoes Sneaker DM '
		'5045–601￼ Flaw',
		'asin': '115456853186'
	}, {
		'Price': ['114.90',
			'119.90'
		],
		'Title': "Air Jordan Stay Loyal Shoes White Black Red DB2884-106 Men's Multi "
		'Size NEW',
		'asin': '155046831159'
	}, {
		'Price': '$8.99',
		'Title': "Harden Volume 3 Men's Basketball Shoes Size 9.5",
		'asin': '175342407862'
	}, {
		'Price': '$59.97',
		'Title': "Men's Nike Precision 5 Basketball Shoes Gym Red Black Grey Bred "
		'Multi Size NEW',
		'asin': '134149634710'
	}]
    assert output == expected

@requests_mock.Mocker(kw="mock")
def test_parse_results_amz(**kwargs):
    # Read mock response data
    mock_file = open("tests/transfer/mocks/mock_parse_results_amz", "rb")
    mock_body = mock_file.read()
    mock_file.close()
    mock_query = "red basketball shoes"
    
    # Invoke function, check response
    query = mock_query.replace(" ", "+")
    kwargs["mock"].get(f"https://www.amazon.com/s?k={query}&page=1", content=mock_body)
    output = parse_results_amz(mock_query, 1)
    expected = [{
		'Price': '59.49',
		'Title': 'High Top Mens Basketball Shoes Lou Williams Streetball Master ' 
			'Breathable Non Slip Outdoor Sneakers Cushioning Workout Shoes for ' 
			'Fitness',
		'asin': 'B083QCWF61'
	}, {
		'Price': '45.99',
		'Title': 'Kids Basketball Shoes High-top Sports Shoes Sneakers Durable '
		'Lace-up Non-Slip Running Shoes Secure for Little Kids Big Kids and '
		'Boys Girls',
		'asin': 'B08FWWWQ11'
	}, {
		'Price': '64.99',
		'Title': 'Unisex-Adult Lockdown 5 Basketball Shoe',
		'asin': 'B0817BFNC4'
	}, {
		'Price': '63.75',
		'Title': 'Unisex-Child Team Hustle D 9 (Gs) Sneaker',
		'asin': 'B07HHTS79M'
	}, {
		'Price': '74.64',
		'Title': 'Unisex-Adult D.O.N. Issue 3 Basketball Shoe',
		'asin': 'B08N8DQLS2'
	}, {
		'Price': '104.90',
		'Title': "Men's Lebron Witness IV Basketball Shoes",
		'asin': 'B07TKMMHVB'
	}, {
		'Price': '36.68',
		'Title': "Unisex-Child Pre-School Jet '21 Basketball Shoe",
		'asin': 'B08N6VRHV4'
	}, {
		'Price': '59.98',
		'Title': "Men's Triple Basketball Shoe",
		'asin': 'B08QCL8VKM'
	}, {
		'Price': '45.98',
		'Title': 'Unisex-Child Pre School Lockdown 4 Basketball Shoe',
		'asin': 'B07HKP12DH'
	}, {
		'Price': '143.72',
		'Title': "Men's Basketball Shoes",
		'asin': 'B07SNR7HRF'
	}]
    assert output == expected

@requests_mock.Mocker(kw="mock")
def test_parse_results_ws(**kwargs):
    # Read mock response data
    mock_file = open("tests/transfer/mocks/mock_parse_results_ws", "rb")
    mock_body = mock_file.read()
    mock_file.close()
    mock_query = "red basketball shoes"
    
    # Invoke function, check response
    query_str = mock_query.replace(" ", "+")
    url = (
        f'{WEBSHOP_URL}/search_results/{WEBSHOP_SESSION}/'
        f'{query_str}/1'
    )
    kwargs["mock"].get(url, content=mock_body)
    output = parse_results_ws(mock_query, 1)
    expected = [{
        'Price': [24.49, 39.99],
        'Title': "BinGoDug Men's Basketball Shoes, Men's Fashion Sneakers, Air "
        'Basketball Shoes for Men, Womens Basketball Shoes, Mens Basketball '
        'Shoes, Boys Basketball Shoes, Youth Basketball Shoes Men Women',
        'asin': 'B09GKFNQWT'
    }, {
        'Price': [1.89, 7.58],
        'Title': "RQWEIN Comfortable Mesh Sneakers Men's Roading Running Shoes "
        'Tennis Shoes Casual Fashion Sneakers Outdoor Non Slip Gym Athletic '
        'Sport Shoes',
        'asin': 'B09BFY2R3R'
    }, {
        'Price': 100.0,
        'Title': 'PMUYBHF Womens Fashion Flat Shoes Comfortable Running Shoes '
        'Sneakers Tennis Athletic Shoe Casual Walking Shoes',
        'asin': 'B09P87V3LZ'
    }, {
        'Price': 100.0,
        'Title': 'PMUYBHF Fashion Travel Shoes Jogging Walking Sneakers Air Cushion '
        'Platform Loafers Air Cushion Mesh Shoes Walking Dance Shoes',
        'asin': 'B09N6SNKC1'
    }, {
        'Price': 100.0,
        'Title': "PMUYBHF Women's Ballet Flats Walking Flats Shoes Dressy Work Low "
        'Wedge Arch Suport Flats Shoes Slip On Dress Shoes',
        'asin': 'B09N6X5S74'
    }, {
        'Price': 100.0,
        'Title': "PWKSELW High-top Men's Basketball Shoes Outdoor Sports Shoes "
        'Cushioning Training Shoes Casual Running Shoes',
        'asin': 'B09MDB9V5W'
    }, {
        'Price': 100.0,
        'Title': "Women's Flat Shoes Classic Round Toe Slip Office Black Ballet "
        'Flats Walking Flats Shoes Casual Ballet Flats',
        'asin': 'B09N6PDFRF'
    }, {
        'Price': 100.0,
        'Title': "Women's Mid-Calf Boots Wide Calf Boots for Women Fashion Zipper "
        'Womens Shoes Pu Leather Casual Boots Womens Slip-On Womens Flat '
        "Shoes Med Heel Womens' Boots Winter Snow Boot Comfy Boots(,5.5)",
        'asin': 'B09N8ZHFNM'
    }, {
        'Price': 100.0,
        'Title': 'PMUYBHF Womens Leisure Fitness Running Sport Warm Sneakers Shoes '
        'Slip-On Mule Sneakers Womens Mules',
        'asin': 'B09P87DWGR'
    }, {
        'Price': 100.0,
        'Title': 'Men Dress Shoes Leather Modern Classic Business Shoes Lace Up '
        'Classic Office Shoes Business Formal Shoes for Men',
        'asin': 'B09R9MMTKR'
    }]
    assert output == expected

def test_convert_dict_to_actions():
    # Test RESULTS page type
    asin = "334490012932"
    page_num = 2
    products = [{
        'asin': '125331076844',
        'Title': 'Modern Tall Torchiere Floor Lamp Brushed Nickel Chrome Metal Decor Living Room',
        'Price': '$129.95'
    }, {
        'asin': '125109985453',
        'Title': 'Floor Lamps Set of 2 Polished Steel Crystal Glass for Living Room Bedroom',
        'Price': '$179.99'
    }, {
        'asin': '125265434055',
        'Title': 'Floor Lamp Nickel/Polished Concrete Finish with Off-White Linen Fabric Shade',
        'Price': '$130.68'
    }, {
        'asin': '195197281169',
        'Title': 'New ListingVintage Mid Century Modern Glass Amber Globe Tension Pole Floor Lamp Light',
        'Price': '$165.00'
    }, {
        'asin': '195197512929',
        'Title': 'New ListingVTG Brass Floor Lamp Glass Shade 63.5" Tall 12" Diameter Glass Shade Original',
        'Price': '$279.45'
    }, {
        'asin': '304550250934',
        'Title': 'Vintage Mid Century Modern 3 Light Tension Pole Floor Lamp glass shades atomic a',
        'Price': '$149.99'
    }, {
        'asin': '175338033811',
        'Title': 'Antique FOSTORIA Ornate Brass Piano  Adjustable Floor Oil Lamp up to 76" Tall !!',
        'Price': '$1,995.00'
    }, {
        'asin': '334490012932',
        'Title': 'Vintage Mid Century Glass Shade Amber Globe 3 Tension Pole Floor Lamp Light MCM',
        'Price': '$128.00'
    }, {
        'asin': '185433933521',
        'Title': 'Brass & Pink Glass Lotus 6 Petal Lamp Shades Set Of Two Replacement Parts As Is',
        'Price': '$90.00'
    }]

    actions = convert_dict_to_actions(Page.RESULTS, products, asin, page_num)

    assert actions['valid'] == [
        'click[back to search]',
        'click[< prev]',
        'click[item - Modern Tall Torchiere Floor Lamp Brushed Nickel Chrome Metal Decor Living Room]',
        'click[item - Floor Lamps Set of 2 Polished Steel Crystal Glass for Living Room Bedroom]',
        'click[item - Floor Lamp Nickel/Polished Concrete Finish with Off-White Linen Fabric Shade]',
        'click[item - New ListingVintage Mid Century Modern Glass Amber Globe Tension Pole Floor Lamp Light]',
        'click[item - New ListingVTG Brass Floor Lamp Glass Shade 63.5" Tall 12" Diameter Glass Shade Original]',
        'click[item - Vintage Mid Century Modern 3 Light Tension Pole Floor Lamp glass shades atomic a]',
        'click[item - Antique FOSTORIA Ornate Brass Piano  Adjustable Floor Oil Lamp up to 76" Tall !!]',
        'click[item - Vintage Mid Century Glass Shade Amber Globe 3 Tension Pole Floor Lamp Light MCM]',
        'click[item - Brass & Pink Glass Lotus 6 Petal Lamp Shades Set Of Two Replacement Parts As Is]'
    ]

    # Test ITEM_PAGE page type
    asin = "224636269803"
    products = {
        '224636269803': {
            'asin': '224636269803',
            'Title': 'Sony SRS-XB01 EXTRA BASS Portable Water-Resistant  Wireless Bluetooth Speaker',
            'Price': '24.99',
            'MainImage': 'https://i.ebayimg.com/images/g/jVEAAOSwCLBhXLuD/s-l500.jpg',
            'Rating': None,
            'options': {
                'Color': ['Black', 'White', 'Red', 'Blue']
            },
            'option_to_image': {},
            'Description': "eBay Sony EXTRA BASS Portable Water-Resistant Wireless Bluetooth SpeakerBRAND NEW ITEMFREE SHIPPING WITHIN USA30 DAY RETURN POLICYKey FeaturesEXTRA BASS for deep, punchy soundCompact portable designUp to 6 hours of battery lifeWater resistant for worry-free useSupplied with color-coordinated strap What's in the Box?Sony EXTRA BASS Portable Bluetooth SpeakerPower supplyUser manual HIGHLIGHTSMUSIC THAT TRAVELSSmall size but mighty in volume to deliver powerful beats wherever you travelHANDS FREE CALLINGWith the built-in microphone, taking calls from your smartphone is easy. SPLASHPROOF CASINGTake to the pool or beach without worrying about water damaging the speaker unit UPGRADE THE AUDIOWirelessly connects 2 speakers and achieve stereo sound with speaker add function LONGER BATTERY LIFELonger Virtual Happy Hours with this rechargeable speaker's 6 hour battery life Technical SpecsFeatureValueBrandSonyTypePortable speakerModel NumberSRSXB01BluetoothYesFrequency range2.4 GHzMax. Communication Range32 ftBattery LifeApprox. 6 hrsWater ProtectionIPX5Input and Output TerminalsStereo Mini Jack (IN)Dimensions (W x H x D)Approx. 3 1/4 × 2 3/8 × 2 1/4 inWeightApprox. 5.65 oz",
            'BulletPoints': "Item specifics Condition:New: A brand-new, unused, unopened, undamaged item in its original packaging (where packaging is ...  Read moreabout the conditionNew: A brand-new, unused, unopened, undamaged item in its original packaging (where packaging is applicable). Packaging should be the same as what is found in a retail store, unless the item is handmade or was packaged by the manufacturer in non-retail packaging, such as an unprinted box or plastic bag. See the seller's listing for full details. See all condition definitionsopens in a new window or tab  Model:EXTRA BASS Connectivity:Bluetooth, Wireless Type:Portable Speaker System Compatible Model:EXTRA BASS, Portable Water-Resistant Features:Bluetooth, Water-Resistant MPN:SRS-XB01/B, SRS-XB01/L, SRS-XB01/R, SRS-XB01/W Brand:Sony"
        }
    }

    actions = convert_dict_to_actions(Page.ITEM_PAGE, products, asin, 1)

    assert actions['valid'] == ['click[back to search]', 'click[< prev]', 'click[description]', 'click[features]', 'click[buy now]', 'click[Black]', 'click[White]', 'click[Red]', 'click[Blue]']

    # Test SUB_PAGE page type
    actions = convert_dict_to_actions(Page.SUB_PAGE, {}, "12345", 1)
    
    assert actions['valid'] == ['click[back to search]', 'click[< prev]']