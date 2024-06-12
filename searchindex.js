Search.setIndex({docnames:["contributing","extending","index","installation","introduction","pipeline","reference"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["contributing.rst","extending.rst","index.rst","installation.rst","introduction.rst","pipeline.rst","reference.rst"],objects:{"renard.graph_utils":[[6,1,1,"","cumulative_graph"],[6,1,1,"","dynamic_graph_to_gephi_graph"],[6,1,1,"","graph_edges_attributes"],[6,1,1,"","graph_with_names"],[6,1,1,"","layout_with_names"]],"renard.ner_utils":[[6,2,1,"","DataCollatorForTokenClassificationWithBatchEncoding"],[6,2,1,"","NERDataset"],[6,1,1,"","_tokenize_and_align_labels"],[6,1,1,"","hgdataset_from_conll2002"],[6,1,1,"","load_conll2002_bio"],[6,1,1,"","ner_entities"]],"renard.ner_utils.DataCollatorForTokenClassificationWithBatchEncoding":[[6,3,1,"","__call__"],[6,3,1,"","__init__"]],"renard.ner_utils.NERDataset":[[6,3,1,"","__init__"]],"renard.pipeline":[[6,0,0,"-","preprocessing"],[6,0,0,"-","stanford_corenlp"]],"renard.pipeline.character_unification":[[6,2,1,"","Character"],[6,2,1,"","GraphRulesCharacterUnifier"],[6,2,1,"","NaiveCharacterUnifier"]],"renard.pipeline.character_unification.Character":[[6,3,1,"","__delattr__"],[6,3,1,"","__eq__"],[6,3,1,"","__hash__"],[6,3,1,"","__init__"],[6,3,1,"","__repr__"],[6,3,1,"","__setattr__"]],"renard.pipeline.character_unification.GraphRulesCharacterUnifier":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","infer_name_gender"],[6,3,1,"","names_are_related_after_title_removal"],[6,3,1,"","needs"],[6,3,1,"","optional_needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.character_unification.NaiveCharacterUnifier":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","needs"],[6,3,1,"","optional_needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.core":[[6,2,1,"","Mention"],[6,2,1,"","Pipeline"],[6,2,1,"","PipelineState"],[6,2,1,"","PipelineStep"]],"renard.pipeline.core.Mention":[[6,3,1,"","__eq__"],[6,3,1,"","__hash__"],[6,3,1,"","__init__"],[6,3,1,"","__repr__"]],"renard.pipeline.core.Pipeline":[[6,4,1,"","PipelineParameter"],[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_non_ignored_steps"],[6,3,1,"","_pipeline_init_steps_"],[6,3,1,"","check_valid"],[6,3,1,"","rerun_from"]],"renard.pipeline.core.PipelineState":[[6,3,1,"","__eq__"],[6,4,1,"","__hash__"],[6,3,1,"","__init__"],[6,3,1,"","__repr__"],[6,4,1,"","char2token"],[6,4,1,"","character_network"],[6,4,1,"","characters"],[6,4,1,"","corefs"],[6,4,1,"","dynamic_blocks"],[6,4,1,"","entities"],[6,3,1,"","export_graph_to_gexf"],[6,3,1,"","get_character"],[6,3,1,"","plot_graph"],[6,3,1,"","plot_graph_to_file"],[6,3,1,"","plot_graphs_to_dir"],[6,4,1,"","quotes"],[6,4,1,"","sentences"],[6,4,1,"","sentences_polarities"],[6,4,1,"","speakers"],[6,4,1,"","text"],[6,4,1,"","tokens"]],"renard.pipeline.core.PipelineStep":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","needs"],[6,3,1,"","optional_needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.corefs":[[6,2,1,"","BertCoreferenceResolver"],[6,2,1,"","SpacyCorefereeCoreferenceResolver"]],"renard.pipeline.corefs.BertCoreferenceResolver":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.corefs.SpacyCorefereeCoreferenceResolver":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_coreferee_get_mention_tokens"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","_spacy_try_infer_spaces"],[6,3,1,"","needs"],[6,3,1,"","optional_needs"],[6,3,1,"","production"]],"renard.pipeline.graph_extraction":[[6,2,1,"","CoOccurrencesGraphExtractor"],[6,2,1,"","ConversationalGraphExtractor"]],"renard.pipeline.graph_extraction.CoOccurrencesGraphExtractor":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_create_co_occurrences_blocks"],[6,3,1,"","_extract_dynamic_graph"],[6,3,1,"","_extract_graph"],[6,3,1,"","needs"],[6,3,1,"","optional_needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.graph_extraction.ConversationalGraphExtractor":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","needs"],[6,3,1,"","production"]],"renard.pipeline.ner":[[6,2,1,"","BertNamedEntityRecognizer"],[6,2,1,"","NEREntity"],[6,2,1,"","NLTKNamedEntityRecognizer"]],"renard.pipeline.ner.BertNamedEntityRecognizer":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","batch_labels"],[6,3,1,"","needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.ner.NEREntity":[[6,3,1,"","__eq__"],[6,3,1,"","__hash__"],[6,3,1,"","__init__"],[6,3,1,"","__repr__"],[6,3,1,"","shifted"],[6,4,1,"","tag"]],"renard.pipeline.ner.NLTKNamedEntityRecognizer":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.preprocessing":[[6,2,1,"","CustomSubstitutionPreprocessor"]],"renard.pipeline.preprocessing.CustomSubstitutionPreprocessor":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.quote_detection":[[6,2,1,"","QuoteDetector"]],"renard.pipeline.quote_detection.QuoteDetector":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.pipeline.sentiment_analysis":[[6,2,1,"","NLTKSentimentAnalyzer"]],"renard.pipeline.sentiment_analysis.NLTKSentimentAnalyzer":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","needs"],[6,3,1,"","production"]],"renard.pipeline.speaker_attribution":[[6,2,1,"","BertSpeakerDetector"]],"renard.pipeline.speaker_attribution.BertSpeakerDetector":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","needs"],[6,3,1,"","production"]],"renard.pipeline.stanford_corenlp":[[6,2,1,"","StanfordCoreNLPPipeline"],[6,1,1,"","corenlp_annotations_bio_tags"]],"renard.pipeline.stanford_corenlp.StanfordCoreNLPPipeline":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","needs"],[6,3,1,"","production"]],"renard.pipeline.tokenization":[[6,2,1,"","NLTKTokenizer"]],"renard.pipeline.tokenization.NLTKTokenizer":[[6,3,1,"","__call__"],[6,3,1,"","__init__"],[6,3,1,"","_pipeline_init_"],[6,3,1,"","needs"],[6,3,1,"","production"],[6,3,1,"","supported_langs"]],"renard.plot_utils":[[6,1,1,"","plot_nx_graph_reasonably"]],"renard.resources.hypocorisms":[[6,2,1,"","HypocorismGazetteer"]],"renard.resources.hypocorisms.HypocorismGazetteer":[[6,3,1,"","__init__"],[6,3,1,"","_add_hypocorism_"],[6,3,1,"","are_related"],[6,3,1,"","get_nicknames"],[6,3,1,"","get_possible_names"]],"renard.utils":[[6,1,1,"","batch_index_select"],[6,1,1,"","block_indices"],[6,1,1,"","search_pattern"],[6,1,1,"","spans"]],renard:[[6,0,0,"-","graph_utils"],[6,0,0,"-","ner_utils"],[6,0,0,"-","plot_utils"],[6,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute"},terms:{"0":6,"0x7fd9e9115900":5,"1":[0,5,6],"10":5,"10000":6,"14":6,"2":[5,6],"20":5,"2002":6,"2014":6,"2015":6,"2017":6,"2019":6,"25":5,"3":[1,5,6],"4":6,"512":6,"639":[1,5,6],"8":6,"8g":6,"9115":6,"9999999":6,"abstract":6,"case":[5,6],"class":[0,1,5,6],"default":[1,5,6],"do":[0,5,6],"export":[5,6],"final":[5,6],"float":6,"function":[0,1,5,6],"import":[1,5],"int":6,"long":5,"new":[0,2,5,6],"return":[0,1,5,6],"static":6,"throw":5,"true":[5,6],"try":6,"while":[5,6],A:[5,6],As:[1,5],By:[1,5],For:[5,6],If:[0,3,5,6],In:[0,5,6],It:5,No:6,One:6,The:[0,2,3,6],These:[4,5],To:5,__call__:[1,5,6],__delattr__:6,__eq__:6,__hash__:6,__init__:[1,6],__repr__:6,__setattr__:6,_add_hypocorism_:6,_context_mask:6,_coreferee_get_mention_token:6,_create_co_occurrences_block:6,_extract_dynamic_graph:6,_extract_graph:6,_must_:6,_non_ignored_step:6,_pipeline_init_:[1,6],_pipeline_init_step:6,_pipeline_init_steps_:6,_spacy_try_infer_spac:6,_tokenize_and_align_label:6,abov:5,accept:6,access:5,accord:6,accordingli:6,ad:6,adapt:6,add:[0,6],addit:6,addition:1,additional_hypocor:6,additional_ner_class:6,after:6,al:6,algorithm:6,alia:[5,6],align:6,all:[0,5,6],alloc:6,allow:[4,5,6],along:6,alreadi:[5,6],also:[0,3,5,6],an:[0,1,2,4,6],analysi:[2,4],analyz:6,ani:[1,5,6],ann:6,annot:[0,5,6],annotate_coref:6,anoth:5,antecedents_nb:6,apach:6,apparit:6,appear:6,appli:[5,6],applic:[0,4],applyfinegrain:6,ar:[0,4,5,6],arbor:6,are_rel:6,arg:6,argument:[0,1,5,6],arrai:6,assign:5,associ:6,attempt:6,attribut:[1,2],auto:6,autom:4,automat:6,avail:2,avoid:6,ax:6,b:6,base:[5,6],basic:[1,6],basictokenizerstep:1,batch:6,batch_i:6,batch_index_select:6,batch_label:6,batch_siz:6,batchencod:6,be_quiet:6,becaus:[5,6],befor:5,begin:6,behind:0,being:6,below:5,bert:6,bert_pipelin:5,bertcoreferenceresolv:5,bertforcoreferenceresolut:6,bertnamedentityrecogn:5,bertspeakerdetector:[5,6],better:0,between:[4,6],biggest:6,bio:6,bio_tag:6,black:0,blob:6,block:[5,6],block_end_index:6,block_indic:[5,6],block_siz:6,block_start_index:6,blocks_indic:5,bool:6,both:6,boundari:6,bypass:6,c:6,call:[1,5,6],callabl:6,can:[0,1,3,4,5,6],cannot:6,carltonnorthern:6,central:5,certain:5,chain:6,chapter:[5,6],char2token:6,charact:[2,4],character_graph:6,character_ner_tag:6,character_network:[5,6],character_unif:[5,6],check:[0,5,6],check_valid:6,choos:6,chosen:6,chunk:6,chunk_siz:6,ci:0,citi:6,client_properti:6,co:6,co_occurences_dist:5,co_occurrences_block:6,co_occurrences_dist:[5,6],code:[1,2,5,6],colab:6,collect:[1,6],com:6,come:6,common:6,comparison:6,compat:6,complet:[0,4],comput:[5,6],concept:5,confer:6,config:6,configur:[5,6],conll2022:6,conll:6,consid:6,consist:0,constant:6,contain:5,context:6,context_mask:6,context_retriev:6,contribut:2,conveni:5,convers:6,conversation_dist:6,conversationalgraphextractor:5,convert:6,cooccurencesgraphextractor:5,cooccurrencesgraphextractor:5,core:[0,1,2],coref:6,coref_model:6,corefer:[2,3],corefere:[3,6],corefereebrok:6,corefre:6,corefs_algorithm:6,corenlp:[2,3],corenlp_annotations_bio_tag:6,corenlp_custom_properti:6,correct:6,correctli:[5,6],correspond:6,costli:0,cpu:6,creat:[0,2,5,6],cuda:6,cumul:6,cumulative_graph:6,current:6,custom:6,customsubstitutionpreprocessor:[5,6],cut:[5,6],cut_into_chapt:5,data:6,datacollatorfortokenclassif:6,datacollatorfortokenclassificationwithbatchencod:6,dataset:6,debug:5,declar:1,def:1,default_quote_pair:6,defin:6,delattr:6,depend:[1,3],deriv:6,descript:6,detail:6,detect:2,determin:[5,6],determinist:6,devic:6,dict:[1,6],dictionari:6,differ:5,dim:6,dimens:6,diminut:6,directli:6,directori:[0,5,6],disabl:6,discard:6,discuss:[0,6],displai:5,distanc:6,distinct:6,doc:[0,6],docstr:0,document:[0,4,5,6],doe:[0,5,6],doesn:5,don:6,done:5,draw:6,dure:5,dweight:6,dynam:[2,6],dynamic_block:[5,6],dynamic_blocks_token:6,dynamic_graph_to_gephi_graph:6,dynamic_overlap:6,dynamic_window:[5,6],e:[3,6],each:[1,4,5,6],earli:6,easili:5,edg:[4,6],eighth:6,either:6,element:6,en:6,encod:[4,6],encount:0,encourag:0,end:6,end_idx:6,eng:[1,6],english:5,entir:[0,6],entiti:2,environ:3,equal:6,error:[5,6],escap:6,et:6,even:6,ever:3,evolv:5,exampl:[1,5,6],except:[5,6],execut:[5,6],exist:0,expect:[5,6],explain:[0,5],explicit:5,explor:5,export_graph_to_gexf:[5,6],extend:2,extra:[3,6],extract:[2,4],extractor:[5,6],f:5,fals:6,featur:[0,5,6],few:5,fig:6,figur:6,file:[0,6],first:[5,6],fledg:5,flexibl:6,follow:[1,3],forget:0,form:6,format:[0,5,6],found:6,four:1,fra:5,french:5,from:[1,4,5,6],from_step:6,frozenset:6,full:[5,6],fulli:5,further:0,futur:6,g:6,gazeet:6,gazett:6,gender:6,gener:6,gephi:[5,6],get:[3,6],get_charact:6,get_nicknam:6,get_possible_nam:6,gexf:[5,6],gilbert:6,github:[0,6],give:0,given:6,global:6,googl:6,graph:[2,4],graph_edges_attribut:6,graph_extract:[5,6],graph_extractor_kwarg:5,graph_start_idx:6,graph_util:6,graph_with_nam:6,graphrulescharacterunifi:5,guidelin:2,ha:6,hack:6,hash:6,hatch:6,have:[4,5,6],head:6,help:5,here:[1,5,6],hgdataset_from_conll2002:6,hierarch:6,hierarchical_merg:6,high:[0,6],hname_const:6,hopefulli:0,howev:[5,6],html:6,http:6,hugginfac:6,hugginface_model_id:6,huggingfac:6,huggingface_model_id:6,humannam:6,hutto:6,hypocorismgazett:6,hypothesi:0,i:6,icwsm:6,id:6,ignor:6,ignored_step:6,implement:[1,6],implemt:6,includ:6,index:[2,6],index_select:6,indic:[5,6],infer:6,infer_name_gend:6,inform:[0,1,5],init:[1,6],initi:6,initialis:6,input:6,insensit:6,inspir:6,instal:[2,6],instead:[5,6],intend:6,interact:[5,6],intern:6,intract:5,introduct:2,intuit:[4,5],invalid:5,io:6,ipynb:6,iso:[1,5,6],issu:[0,5,6],iter:6,its:[5,6],j:6,joshi:6,june:6,keep:6,kei:6,kept:6,know:6,kwarg:[1,6],labatutandbost2019:4,label:6,label_all_token:6,lang:[5,6],languag:[1,5,6],last:6,layout:6,layout_nx_graph_reason:6,layout_with_nam:6,least:1,lee:6,length:6,let:5,letter:6,level:[0,6],leverag:5,librari:[5,6],licens:6,lifetim:6,limit:6,line:6,link:6,link_corefs_ment:6,list:[5,6],liter:6,live:0,load:6,load_conll2002_bio:6,local:0,locat:6,longest:6,lookup:6,loos:6,lot:6,m:[0,6],made:6,mai:[5,6],maintain:0,make:[0,5],manag:3,manual:[2,5,6],map:6,mask:6,master:6,match:[0,6],mathemat:4,matplotlib:[5,6],max:6,max_char_length:6,max_chunk_s:6,max_len:6,max_span_s:6,maximum:6,mean:6,meanwhil:5,media:6,memori:6,mention:[0,5,6],mention_head:6,mentions_per_token:6,merg:[0,6],messag:5,method:[1,5,6],mi:6,might:6,min_appear:[5,6],minimum:6,misc:6,model:[5,6],modifi:6,modul:[0,2,5],more:[4,5,6],most_frequ:6,ms:6,multilingu:2,multipl:5,must:[1,6],my_doc:5,my_doc_in_french:5,my_script:3,my_tokenization_funct:5,n:6,naivecharactersextractor:5,naivecharacterunifi:5,name1:6,name2:6,name:2,name_styl:6,names_are_related_after_title_remov:6,narr:4,ndarrai:6,necessari:0,need:[1,6],neeed:6,ner:5,ner_ent:6,ner_util:6,nercontextretriev:6,nerdataset:6,nerent:6,network:[4,5,6],networkx:[5,6],neural:6,newlin:6,next:6,nicknam:6,nlp:[5,6],nltk:[5,6],nltknamedentityrecogn:5,nltksentimentanalyz:5,nltktoken:5,nnp:6,node:[4,6],non:0,none:6,normal:6,note:[5,6],notebook:6,novel:6,now:6,number:[5,6],nx:6,object:[4,5,6],occur:[5,6],occurr:6,onc:0,one:[5,6],ones:0,onli:[5,6],onlin:0,open:[0,5,6],option:[1,6],optional_ne:[1,6],order:[5,6],org:6,organ:6,origin:6,other:6,otherwis:6,our:0,out:[5,6],output:[2,6],overlap:6,overrid:5,overridden:[1,6],overriden:6,overview:[0,2,4],pad_to_multiple_of:6,page:2,param:6,paramet:[5,6],parsimoni:6,part:[5,6],partial_match:6,particular:[0,6],pass:[0,1,5,6],patch:0,path:6,pattern:6,per:6,perform:[5,6],person:6,pip:2,pipelin:[1,2,3,4],pipelineparamet:6,pipelinest:[1,5,6],pipelinestep:[1,5,6],platform:5,plot:5,plot_graph:[5,6],plot_graph_to_fil:[5,6],plot_graphs_to_dir:[5,6],plot_nx_graph_reason:6,plot_util:6,plt:5,po:6,poetri:[3,6],polar:6,posit:6,possibl:[0,4,5,6],pre:6,preconfigur:5,predict:6,prefix:6,preprocess:2,preprocessor:6,pretrainedmodel:6,pretrainedtokenizerfast:6,previou:[5,6],previous:6,problem:0,produc:[1,6],product:[1,6],progress:6,progress_report:6,progressreport:6,project:3,pronoun:6,propag:5,properti:6,provid:5,pull:0,py:3,pyplot:5,pytest:0,python:[0,3,4,6],pytorch:6,qualiti:2,quot:2,quote_detect:6,quote_pair:6,quotedetector:5,r:6,ram:6,rather:[5,6],rational:0,re:6,read:[5,6],readm:6,reason:6,recogn:6,recognit:2,recomput:6,record:6,refer:[0,2],regardless:5,regex:[5,6],regroup:5,relabel:6,relat:6,relationship:[4,6],relev:[0,6],reli:[0,5],remov:6,renard:[0,3,4,5,6],renard_test_al:0,report:6,repositori:0,repr:6,repres:[4,5],representend:5,request:0,requir:[5,6],rerun_from:6,research:6,resolut:2,resolv:[3,6],resolve_inconsist:6,resort:6,resourc:2,result:[5,6],retriev:6,richardpaulhudson:6,rst:0,rule:6,run:[0,1,3,5,6],runtim:6,runtm:6,s:[0,5,6],same:[1,5,6],satisfi:[5,6],satisifi:0,save:[5,6],script:3,scrollto:6,search:[2,6],search_pattern:6,second:6,see:[4,6],seen:5,select:6,self:[1,6],sentenc:[5,6],sentences_polar:6,sentiment:2,sentiment_analysi:6,separ:6,seq:6,sequenc:6,sequenti:[5,6],seri:6,server:6,server_kwarg:6,server_timeout:6,set:[1,5,6],setattr:6,sever:[4,5,6],shall:6,shape:6,share:6,shell:3,shift:6,shortest:6,should:[0,1,5,6],show:[0,5],simpl:[5,6],simpli:3,simplic:5,sinc:6,singl:[4,6],size:6,slider:[5,6],smallest:6,so:[0,6],social:6,solver:3,some:6,sometim:6,sourc:0,space:6,spaci:[3,6],spacycorefereecoreferenceresolv:5,span:6,speaker:2,speaker_attribut:6,special:[1,5],specif:[0,1],specifi:[1,5,6],spinx:0,split:[1,6],spuriou:6,stable_layout:6,stai:0,stanford:[2,3],stanford_corenlp:6,stanfordcorenlppipelin:[5,6],stanfordnlp:6,stanza:[3,6],start:6,start_idx:6,state:[1,2],statist:6,step:2,still:6,store:6,stori:4,str:[1,6],string:[1,5,6],strongest:6,style:[0,6],substit:6,substition_rul:6,substitut:[5,6],support:[1,2,6],supported_lang:[1,5,6],suppos:5,sure:0,t:[5,6],tag:6,tag_conversion_map:6,task:5,tensor:6,termin:6,test:0,text:[1,4,5,6],than:[5,6],thei:[4,6],them:[5,6],therefor:5,thi:[1,5,6],thing:5,those:[5,6],though:6,through:5,tibert:[5,6],time:[0,1,5,6],timeout:6,timestep:6,titl:6,token:[1,2],token_classif:6,tool:[4,5],torch:6,tqdm:6,trade:6,train:6,transform:6,trivial:0,troubleshoot:5,tupl:6,turn:6,two:[4,6],txt:5,type:[0,1,6],typevar:6,under:[3,6],unic:6,unif:2,unifi:6,union:6,uniqu:5,unit:6,unknown:6,up:[0,5,6],us:[0,1,2,4,5,6],usag:6,usual:[1,5,6],util:[2,5],vader:[5,6],vala:6,valid:[1,5,6],valu:[1,5,6],variabl:6,vc0bsbliirjq:6,version:6,virtual:3,visual:5,visualis:[4,5],wa:[5,6],wai:[1,6],want:[0,3,5],warn:6,we:0,weblog:6,weight:6,weirdli:6,welcom:0,well:6,when:[0,5,6],where:[4,6],which:[5,6],whole:6,why:[5,6],wide:1,window:[5,6],wise:6,wish:6,within:6,without:6,wont:6,wordpiec:6,work:5,would:5,wp_label:6,write:0,write_gexf:6,written:4,yet:6,york:6,you:[0,3,5,6],your:[0,5],yourself:[0,5]},titles:["Contributing","Extending Renard","Welcome to Renard\u2019s documentation!","Installation","Introduction","The Pipeline","Reference"],titleterms:{"new":1,The:5,an:5,analysi:[5,6],attribut:[5,6],avail:5,bertcoreferenceresolv:6,bertnamedentityrecogn:6,charact:[5,6],code:0,content:2,contribut:0,conversationalgraphextractor:6,cooccurrencesgraphextractor:6,core:6,corefer:[5,6],corenlp:6,creat:1,custom:5,detect:[5,6],document:2,dynam:5,entiti:[5,6],extend:1,extract:[5,6],graph:[5,6],graphrulescharacterunifi:6,guidelin:0,hypocor:6,indic:2,instal:3,introduct:4,manual:3,multilingu:5,naivecharacterunifi:6,name:[5,6],ner:6,nltknamedentityrecogn:6,nltksentimentanalyz:6,nltktoken:6,output:5,overview:5,pip:3,pipelin:[5,6],plot:6,preprocess:[5,6],qualiti:0,quot:[5,6],quotedetector:6,recognit:[5,6],refer:6,renard:[1,2],resolut:[5,6],resourc:6,s:2,segment:5,sentiment:[5,6],spacycorefereecoreferenceresolv:6,speaker:[5,6],stanford:6,state:[5,6],step:[1,5,6],support:5,tabl:2,token:[5,6],unif:6,us:3,util:6,welcom:2}})