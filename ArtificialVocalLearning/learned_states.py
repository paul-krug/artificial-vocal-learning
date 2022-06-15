




states = {
	'a': [0.44888324353056513, -4.12235424309943, -0.5, -7.0, -0.7203599206332302, 1.3286646809074663, 0.3580593402235236, -0.1, -0.5699015039391692, -1.624550001596718, 3.349780630792794, -3.0, -2.8703063798224293, 1.7209615101034457, -1.3406, -2.1366, 0.04415078280489609, 0.0, -0.9187590945779434],
	'e': [1.0, -5.157127526508459, -0.2095329078788178, -6.89276706846005, -0.00255020492690255, 0.6621860760140783, 0.0, -0.1, 4.0, -0.8266699166867296, 1.5055533464636333, -1.67169010023017, -1.318155549486661, 0.39466899662157545, -1.3406, -2.1366, 0.09054381495020146, 0.028695043736777535, -0.6217443130010414],
	'i': [0.9107705749075836, -5.770634476535357, -0.3821535296855989, -3.7389725413984327, -0.47707612722423226, 1.1804842453313567, 0.7954887907603526, -0.1, 2.2202169526066506, -0.7595552223540789, 3.303397144825327, -1.6560495302198597, 3.625237501141089, 1.1135601513083377, -1.3406, -2.1366, 0.2762518213459323, 0.2732977748288777, 0.9660545562170453],
	'o': [0.4492285703164862, -5.216288412964081, -0.43688003014979826, -6.365987166898982, 0.5426127484198944, 0.2612588834336996, 0.7023164474461887, -0.1, -0.6231120780273532, -1.5826946954671153, 1.590481689057611, -3.0, -2.439242043925953, 0.17596818795653488, -1.3406, -2.1366, 0.02224315968817721, 0.0, -0.9698065373616083],
	'u': [0.988672527479577, -5.05304492742647, -0.4987069992675551, -5.88746160066637, 1.0, 0.356374145784502, 0.18071723189965513, -0.1, -0.7483120553853687, -2.8455543713987335, 1.5723035070886608, -2.8104111793664583, -2.4857284960267703, -0.8195947780450227, -1.3406, -2.1366, 0.10550946747996216, 0.0, -0.9625367546587513],
	'E': [0.006567996515018231, -3.5424060202169447, -0.49908374443437187, -5.890476188882848, -0.809791149071537, 1.9132048364297791, 0.6904442964072158, -0.1, 2.2051446411866715, -1.9124331874113794, 5.378831269946111, -2.3392322989878656, 0.008078055093829195, 2.528510975566656, -1.3406, -2.1366, 0.927986033029741, 0.906368105807845, -0.1983935402928883],
	'I': [0.9484334629162936, -3.9821670163157865, -0.18920267303921362, -2.2934999976062453, 1.0, 1.1556351443210826, 0.015904438006987005, -0.1, 2.460569778014664, -2.2228396216492023, 3.162669681250302, -2.239235946971382, -2.237553679720437, -2.237768896304357, -1.3406, -2.1366, 0.0, 1.0, -0.746411982323794],
	'O': [0.5473059176871442, -3.5, -0.47859763686156215, -5.295322247298256, 0.8446858276059027, 1.563475852023613, 1.0, -0.1, 0.43258640177694496, -1.4049633336217862, 3.3653557808724974, -2.684766384201409, -2.6586384284356512, 1.6263230038911993, -1.3406, -2.1366, 0.0, 0.6472065757939657, -0.9542867873355756],
	'U': [0.7602902497429861, -5.696744343194806, -0.5, -6.9468915776782065, 0.2951110125174697, 0.149617275628768, 0.24613393206588002, -0.1, 0.05376238696162228, -1.4590797471634416, 1.5, -1.5631781039720325, -1.595934383112829, -3.0, -1.3406, -2.1366, 0.8762697569979662, 0.44098398598560024, -0.4175232076080526],
	'2': [0.0, -4.73071341759626, -0.23054161391516684, -3.2380405678051534, 0.9709523340110562, 0.5097344276103082, 0.15731655666011748, -0.1, 1.3243354550333446, -2.9752088977187428, 1.5936326404198213, -0.5440578647588892, 4.0, -1.4077548884299498, -1.3406, -2.1366, 1.0, 0.43050436957859584, 0.444701143940085],
	'9': [0.1491716858972778, -6.0, -0.49683957715733335, -6.960353455660962, 0.6383366555605703, 0.43842091852458853, 0.0, -0.1, 1.3687125103525057, -2.0765811445380065, 2.9565705170277057, -1.166075495181417, 3.6976420016652236, -1.0322359468212348, -1.3406, -2.1366, 0.4825828733672421, 0.5942774267826691, 0.1342404325946256],
	'y': [1.0, -6.0, -0.22788926649623345, -3.3732670369646285, 0.9368759934972362, 1.118080258615152, 0.0005859351256179562, -0.1, 2.4661733634554244, -1.0400449252977249, 1.5, -2.990014239165512, -0.39841455424498484, -3.0, -1.3406, -2.1366, 0.0, 0.4455811221167655, -0.8293303686841744],
	'Y': [0.0, -5.4548363224926195, -0.4936500423138235, -5.553785888142224, 1.0, 0.6033953762040511, 0.5994005398728888, -0.1, 2.2545401289798694, -1.6392652001055934, 2.082780489980885, -2.7563475364264542, 1.656611630310325, -1.8899219143391557, -1.3406, -2.1366, 0.0, 1.0, 0.06596739009926872],
	'@': [0.45452008096192553, -6.0, -0.5, -3.434347640798292, -0.2426333366912315, 0.8065414868375012, 0.291732248340821, -0.1, 1.7635808429060185, -3.0, 3.0042237267959955, -3.0, -2.6669080468918054, -3.0, -1.3406, -2.1366, 0.034494491451647624, 0.18359686916802864, -0.9486257693064722],
	'6': [0.4599491754785736, -5.251826743418837, -0.4756925652153041, -2.951687698109289, 0.8897003418618592, 1.9074746132608669, 1.0, -0.1, 0.42587912690562263, -2.679159100837354, 5.5, -2.502189550145681, -1.3978569915083132, 1.371102641993874, -1.3406, -2.1366, 0.0, 1.0, -0.9370201900287822],
}


visual_states = {
	'a': [0.8595036947872966, -4.351481994313003, -0.00031592152712055764, -5.674585918636518, 0.17265511100048772, 1.2022741194549955, 0.30822660834980536, -0.1, 0.5612233350541752, -3.0, 1.897396436882445, -0.09664912265053809, -0.19509024914777984, 2.1949724511383186, -1.3406, -2.1366, 0.718010854150614, 0.9918326020678766, 0.33057823597707325],
	'e': [0.3022978602645637, -5.030640563081231, -0.0029057411539501424, -2.1473708050725056, -0.4185159229970148, 0.5872910499353601, 0.10401394291175622, -0.1, 2.533836140425642, -1.0639471704830838, 3.6516164908694977, -2.1854267100103852, 1.4441003437227802, 5.0, -1.3406, -2.1366, 0.019728552520961947, 0.7865099254002982, -0.4551878949967236],
	'i': [0.9974823718249421, -5.470698878713803, -0.043085272561666395, -1.9317168277389807, -0.2869856361040539, 0.5021080629227553, 0.9981081113957461, -0.1, 2.6930324150881337, -0.6981820342693553, 2.8063794195823974, -2.821871507829899, 3.0987391197434975, -2.9994827622243663, -1.3406, -2.1366, 0.17947060993547653, 1.0, 0.995406154765157],
	'o': [0.030031155962634802, -4.171670243768962, -0.09857222320342811, -2.289392576129105, 0.6371931605620834, 0.30977731404362885, 0.008959475050276248, -0.1, -0.4386525369312627, -1.0783217462383818, 1.5808677074697233, -2.980034094705902, -2.8507841932108153, -2.1800661855875534, -1.3406, -2.1366, 0.6056653455693766, 0.0, -0.25791696098891165],
	'u': [0.9918476932621255, -4.934181005356306, -0.19484971213063562, -1.3375088301923768, 0.6006400636023427, 0.27267200061161384, 0.0012121621004064706, -0.1, -0.5617471839436903, -2.8142270691565976, 1.6900021332066413, -2.114901075068527, 0.5706683870179469, -1.9395975621922368, -1.3406, -2.1366, 0.9156693166287521, 0.0029646632541012316, 0.26011484514453803],
	'E': [0.9965134334919423, -3.5, -0.49906778526381906, -5.725340467327071, -0.11646565369118488, 1.1581803611690256, 0.3152420945790869, -0.1, 1.441137662886311, -1.264176548340468, 4.96929589605816, -2.9755262218677956, 3.489169640244536, 1.333899287599186, -1.3406, -2.1366, 0.8162049674834275, 0.0, 0.1008854201492367],
	'2': [0.2801881069036356, -5.331361659330575, -0.20113808501064487, -1.4068784771782867, 0.38032717405677124, 0.3533703052266244, 0.02534336470582343, -0.1, 1.4389897865595374, -2.2798347470437585, 1.5, 0.7201117270290529, -0.21339525950152105, 0.2332967757100725, -1.3406, -2.1366, 0.022432738064686104, 0.5591130306028157, -0.3185291015308393],
	'y': [0.9274983326220861, -5.743390734080141, -0.27503731389385183, -1.4351430851068705, 0.4756496363603889, 0.5269393656888789, 0.0, -0.1, 2.448563927346709, -0.9175224252308374, 1.6169110797390092, -1.4034091098272015, -0.19772873490935683, 2.6983033752126, -1.3406, -2.1366, 0.1547359434675147, 0.059160752526385155, -1.0],
}



def get_learned_state( phoneme, include_visual_information ):
	if include_visual_information:
		return states[ phoneme ] #visual_states[ phoneme ]
	else:
		return states[ phoneme ]