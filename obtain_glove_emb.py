from utils import *

now = datetime.now()
date_tag = now.strftime("%Y%m%d")

semcor = False
masc = True

save = True

save_str = 'semcor'

fname = f'dfs_{save_str}_all_20211203.pkl'


if __name__ == "__main__":
	df = pd.read_pickle(fname)
	
	# Create a df version with dropped rows according to empty rows in words_no_punc
	df_no_punc = df.drop(df[df.words_no_punc == ''].index)
	
	if save:
		df_no_punc.to_pickle(f'dfs_{save_str}_all_no_punc_{date_tag}.pkl')
	
	# Obtain all unique words
	vocab = set(df_no_punc.words_no_punc_w_lemma_critical_word.str.lower())
	# vocab = {'wedding', 'camp'}
	
	# Get GloVe dict for words of interest
	w2v = read_glove_embed(vocab, GLOVEDIR + 'glove.840B.300d.txt')
	print(f'The following {len(vocab - w2v.keys())} words not available in GloVe dict: {vocab - w2v.keys()}')  # missing words
	
	if save:
		w2v_df = pd.DataFrame.from_dict(w2v, orient='index')
		w2v_df.to_pickle(f'w2v_df_{date_tag}.pkl')
	
	lst_glove_emb = []
	for row in df_no_punc.itertuples():
		word_of_interest = row.words_no_punc_w_lemma_critical_word.lower()
		if word_of_interest in w2v:
			glove_embed = w2v[row.words_no_punc_w_lemma_critical_word.lower()]
		else:
			glove_embed = np.empty(300)
			glove_embed[:] = np.nan
		lst_glove_emb.append(glove_embed)
	
	df_no_punc['glove_emb'] = lst_glove_emb
	
	if save:
		df_no_punc.to_pickle(f'dfs_{save_str}_all_no_punc_w_glove_emb_{date_tag}.pkl')
		



# na_words = ['selfimages', 'boatel', 'yancey6', 'pitrun', 'ruandaurundi', 'floortoceiling', 'nowmisplaced', 'gordin', 'podger', 'completelyrestored', 'dipylon', 'tworecord', 'grandlooking', 'odwyer', 'couperin', 'reuveni', 'europeanized', 'radarcontrolled', 'rundfunk', 'unfunnily', 'jarrodsville', 'radiochlorine', 'leninismmarxism', 'appestat', 'midjune', 'subchiefs', 'doubleglaze', '37679', 'overfall', 'informationseekingsense', 'nearmisses', 'safavids', 'grappely', 'hohlbein', 'formcreating', 'laysisters', 'mitropoulos', 'jerebohms', 'hundredandfifty', 'drawingrooms', 'ratface', 'badrawi', 'deslonde', '1080062', 'gizenga', 'exfighter', 'torrio', 'heidenstam', 'momoyama', 'dannehower', 'mothersinlaw', 'skolovsky', 'globocnik', 'anotherthe', 'expresident', 'gaafer', 'sopsaisana', 'dancetheatre', 'grabski', 'halfman', 'duclos', 'wellequipped', 'parrillo', 'stansbery', 'artkino', 'amorist', 'redblooded', 'wycoff', 'cherkasov', 'rockcarved', 'drygulchin', 'wonlost', 'burle', 'carnegey', 'lindemanns', 'wellkept', 'reavey', 'intrastellar', 'kulturbund', 'lesourd', 'bylot', 'dellwood', 'crazywonderful', 'musicloving', 'buzzbuzzbuzz', 'seato', 'cottongrowing', 'saintsaens', 'triandos', 'thuggee', 'manybodied', 'depugh', 'czerny', 'brainwracking', 'flannagan', 'muledrawn', 'earnedrun', 'hardwickeetter', 'selfdeception', 'selfrealized', 'photofloodlights', 'cliburn', '795586', 'sevenweek', 'postattack', 'kabalevsky', 'centerpunch', 'onestroke', 'thirtyfour', 'gorboduc', 'dresbachs', 'toohearty', 'macropathological', 'vienot', 'gursel', 'welladvised', 'matinals', 'wellcemented', 'lubra', 'lagerlo', 'marsicano', 'highceilinged', 'camusfearna', 'aricaras', 'banion', 'ugf', 'fairsized', 'selfsustaining', 'gapt', 'communese', 'targo', 'longbodied', 'sweetsour', 'facesaving', '344000', 'charlayne', 'schaack', 'matunuck', 'ekstrohm', 'balaguer', 'burntred', 'swearingin', 'whitestucco', 'nordmann', 'odilo', 'taraday', 'shoettle', 'interama', 'pridestarlette', 'ttau', 'nerveshattering', '37470', 'anticatholicism', 'menilmontant', 'tenyearold', 'grinsfelder', 'selfcriticism', 'noncatholic', 'meyner', 'earthweeks', 'waterfilled', 'odwyers', 'bmews', 'cranelike', 'fiveply', 'jannequin', 'oystchers', 'siddo', 'gontran', 'globedemocrat', 'fivefoot', 'szelenyi', 'halfmelted', 'wellfed', 'clerfayt', 'goldphone', 'dresbach', 'afterduty', 'grillework', 'gauleiter', 'sepulchred', 'yachtel', 'hemus', 'talleyrand', 'inchthick', 'mullenax', 'purdew', 'cochairmen', 'halfinch', 'thirtynine', 'kegham', 'outofstate', 'hardliquor', 'andrenas', 'crosspurposes', 'singlefoot', 'lightcolored', 'finefeathered', 'mattathias', 'nyberg', 'agnese', 'prelegislative', 'expansioncontraction', 'chargeexcess', 'leale', 'dirion', 'heavyarmed', 'skolman', 'vermeersch', 'manysided', 'wratten', 'mathues', 'pittsburghers', 'mohammedanism', 'kupcinet', 'templeman', 'blastdown', 'buckra', 'piraro', 'macneff', 'weigand', 'juet', 'shayol', 'benedick', 'broglio', 'strongrooms', 'matamoras', 'shahn', 'handscreened', 'jastrow', 'toolarge', 'bigshouldered', 'killingsworth', 'creekturn', 'rickenbaugh', 'petipa', 'torridmighty', 'gilborn', 'barnaba', 'norell', 'virsaladze', 'stardel', 'wellwritten', 'yachtels', 'onestory', 'selfeffacement', 'oistrakh', 'boltaction', 'columnshaped', 'heavyelectricalgoods', 'hanoverbertie', 'selfdeceiving', 'kohnstamm', 'advertisingconscious', 'onthejob', 'thynne', 'designconscious', 'glocester', 'costdata', 'prokofieff', 'supercondamine', 'selkirks', 'plasticcovered', 'beardens', 'flannagans', 'quarterinch', 'nonsegregated', 'capandball', 'ballestre', 'franksinbuns', 'allwoman', 'halfcentury', 'nomias', 'katanga', 'bizerte', 'sukarno', 'lauchli', 'coalblack', 'hirey', 'hultberg', 'newburger', 'soignee', 'knobbyknuckled', 'salesconscious', 'hengesbach', 'seigner', 'shotguntype', 'sx21', 'manderscheid', 'rollsroyce', 'fortysix', 'guardino', 'besttempered', 'secretarydesignate', 'besset', 'thiihng', 'distortable', 'bangsashes', 'acoming', 'athabascan', 'bathyran', 'mailedfistinvelvetglove', 'boxell', 'ncta', 'barcus', 'jolliffe', 'jelke', 'maecker', 'steamily', 'edmonia', 'collectivebargaining', 'nisfijahan', 'weldwood', 'lowwater', 'bestlooking', 'dalzellcousin', 'ablard', 'abernathys', 'fitc', 'nogol', 'shortrun', 'prayertime', 'brandel', 'moderndance', 'afrocuban', 'schockler', 'iodocompounds', 'admassy', 'ierulli', 'kapnek', 'dromozoa', 'gasfired', 'mccloy', 'cumbancheros', 'inquisitorgeneral', 'theatrebythesea', 'bastianini', 'wisman', 'sonenberg', 'counterdrill', 'welladministered', 'toonaked', 'baldrige', 'familycommunity', 'hammarskjold', 'onethousandzloty', 'goodis', 'chadroe', 'todman', 'bakeoffs', 'klauber', 'rifleshotgun', 'firzite', 'kililngsworth', 'oneinch', 'railmobile', 'whipsnade', 'luxer', 'onleh', 'arteriolosclerosis', 'boatels', '80738', 'kodyke', 'merner', 'pricecutting', 'lummus', 'mennen', 'yearearlier', 'customdesign', 'lightreflecting', 'carmer', 'rossoff', 'threemonth', 'gershwins', 'privateeye', 'doaty', 'nonitemized', 'ffold', 'kyne', '7360187', 'paot', 'goyette', 'misperceives', 'lolotte', 'mazeroski', 'teewah', 'differentcolor', 'tawes', 'kemm', 'nitrogenmustard', 'skyros', 'fiscaltax', 'branum', 'wollman', 'mclish', 'wwrl', 'newbiggin', 'mahzeer', 'tewfik', 'seconddegree', 'hundredodd', 'kawecki', 'offthecuff', 'sarasate', 'hendl', 'zemlinsky', '400000000', 'garryowen', 'craftindustrial','image002gif', 'pp118', 'areif', 'wsj0032txt', 'drmcwealthyahoocojp', '2347098801668', 'npoppelierelseviernl', 'httptinyurlcomcsqcng', 'athleticstxt', '45723', 'napulcans', 'keerthisingam', 'gamino', 'wwwamazonfailcom', '10t194033z', 'bhaskara', 'muise', 'libertyrant', 'weigle', 'arletta', 'httptinyurlcomd658mv', 'koleniko', 'undelighted', 'lightscreen', '40dn', '012208', 'iiir', 'vaupes', 'scene5txt', 'pintel', 'urdang', 'ctbo', 'delian', 'httpbitlydopc6', 'tokhtakhounov', 'httpstwittercomjilledesigns', 'mindbusnl', 'dobbens', 'ericholdeman', '11t100042z', 'e174', '09t150951z', 'httpbitlyssoq', 'florafauna', 'semimanual', 'almightyus', '�', 'digoxygenin', 'cadiac', 'teapublicans', 'ninjen', '713853', 'microstrength', 'yhl', 'greyhem', 'kyoga', 'antiena', 'rollingstonecom', 'arizonians', 'instensly', 'uwax5', 'ragetti', 'bankwest', 'recommendationdoc', 'medpacs', 'hhrs', '10t203424z', 'tarrin', '2348059781980', 'httpcaselawlpfindlawcomscriptsgetcaseplcourtusvol000invol04108kelo', 'chapelwas', '10t153925z', 'suchoff', 'oilstates', 'lexicological', 'bnewton', 'httpbitlyf4gnc', 'aggi', 'httpcitizensvoicecomartsliving22122213fredfranziawinesarchiebunkerwithadoseofdonaldtrump1564510', 'pendlebury', 'robomole', 'judisch', 'fiancèe', 'hellyeahdude', 'inversional', 'httpbitlyqc5n', 'crossexamination', 'chaperming', '45815', 'indianness', 'hermannsburg', 'scottdoc', 'acr4', 'southoz', 'lovinglife307', 'httptinyurlcomcktpo7', 'republicander', 'smithills', 'coloane', '073905', 'httptinyurlcomc49rgl', 'image001png', 'issuethanks', '25jul02txt', 'sorna', 'gonif', 's218', 'hendrix6406', 'reasor', 'illbyrneeliteee', 'cogdogroo', 'nigeriaresolutionpanelpresidencycom', 'weick', 'noncatholic', 'perdiew', 'eeeerrrrrrrrr', 'godfred', '4100n', 'ipray', 'carrère', 'httptinyurlcom38mqvn', 'httpbitlyloptx', 'njcom', '45836', 'workinga', 'retromodo', 'kookoóee', 'httpbitly5j6o', 'montedison', '4853482', 'pronou', 'jurassicparkiv', 'nepthys', 'marzark', 'molenwerf', 'moresuccesful', 'norrington', 'johnlopresti', 'mmmmmmmmmmmmmmmmdairyqueen', 'kishori', 'nafis', '45849', 'luvsrocket2themoon', 'lasetjet', 'bartók', 'acr3', 'gernisht', '37celsius', '45792', 'httpwwwfacebookcomjilledesigns', '10t202343z', 'iím', 'bowli', '111348txt', '45791', 'poppelier', 'sanuzis', 'rightgirl', 'tambala', '247267102', 'stlp', 'elding', 'cousinavi', '10t165133z', 'mc99', 'httppctmicrosoftcomstlp', 'macroweaknesses', 'bugdave', 'fightn', 'jcfundrzraolcom', 'eitc', 'wendyshanahan', 'pelegostos']
