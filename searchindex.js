Search.setIndex({docnames:["index","installation","pipeline","resources"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","installation.rst","pipeline.rst","resources.rst"],objects:{"renard.pipeline":[[2,0,0,"-","characters_extraction"],[2,0,0,"-","graph_extraction"],[2,0,0,"-","ner"],[2,0,0,"-","preprocessing"],[2,0,0,"-","stanford_corenlp"],[2,0,0,"-","tokenization"]],"renard.pipeline.characters_extraction":[[2,1,1,"","Character"],[2,1,1,"","GraphRulesCharactersExtractor"],[2,1,1,"","NaiveCharactersExtractor"],[2,3,1,"","_assign_coreference_mentions"]],"renard.pipeline.characters_extraction.Character":[[2,2,1,"","__delattr__"],[2,2,1,"","__eq__"],[2,2,1,"","__hash__"],[2,2,1,"","__init__"],[2,2,1,"","__repr__"],[2,2,1,"","__setattr__"]],"renard.pipeline.characters_extraction.GraphRulesCharactersExtractor":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","infer_name_gender"],[2,2,1,"","needs"],[2,2,1,"","optional_needs"],[2,2,1,"","production"]],"renard.pipeline.characters_extraction.NaiveCharactersExtractor":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","optional_needs"],[2,2,1,"","production"]],"renard.pipeline.core":[[2,1,1,"","Pipeline"],[2,1,1,"","PipelineState"],[2,1,1,"","PipelineStep"]],"renard.pipeline.core.Pipeline":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","check_valid"]],"renard.pipeline.core.PipelineState":[[2,2,1,"","__eq__"],[2,4,1,"","__hash__"],[2,2,1,"","__init__"],[2,2,1,"","__repr__"],[2,4,1,"","bert_batch_encoding"],[2,4,1,"","bio_tags"],[2,4,1,"","chapter_tokens"],[2,4,1,"","chapters"],[2,4,1,"","characters"],[2,4,1,"","characters_graph"],[2,4,1,"","corefs"],[2,2,1,"","draw_graph"],[2,2,1,"","draw_graph_to_file"],[2,2,1,"","draw_graphs_to_dir"],[2,2,1,"","export_graph_to_gexf"],[2,2,1,"","graph_with_names"],[2,4,1,"","text"],[2,4,1,"","tokens"],[2,4,1,"","wp_bio_tags"],[2,4,1,"","wp_tokens"]],"renard.pipeline.core.PipelineStep":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","optional_needs"],[2,2,1,"","production"]],"renard.pipeline.corefs":[[2,0,0,"-","bert_corefs"],[2,0,0,"-","corefs"]],"renard.pipeline.corefs.bert_corefs":[[2,1,1,"","BertCoreferenceResolutionOutput"],[2,1,1,"","BertForCoreferenceResolution"],[2,1,1,"","CoreferenceDataset"],[2,1,1,"","CoreferenceDocument"],[2,1,1,"","DataCollatorForSpanClassification"],[2,3,1,"","load_wikicoref_dataset"]],"renard.pipeline.corefs.bert_corefs.BertCoreferenceResolutionOutput":[[2,2,1,"","__eq__"],[2,4,1,"","__hash__"],[2,2,1,"","__init__"],[2,2,1,"","__repr__"]],"renard.pipeline.corefs.bert_corefs.BertForCoreferenceResolution":[[2,2,1,"","__init__"],[2,2,1,"","bert_parameters"],[2,2,1,"","closest_antecedents_indexs"],[2,2,1,"","forward"],[2,2,1,"","loss"],[2,2,1,"","mention_compatibility_score"],[2,2,1,"","mention_score"],[2,2,1,"","predict"],[2,2,1,"","pruned_mentions_indexs"],[2,2,1,"","task_parameters"]],"renard.pipeline.corefs.bert_corefs.CoreferenceDataset":[[2,2,1,"","__init__"],[2,2,1,"","from_conll2012_file"],[2,2,1,"","merged_datasets"]],"renard.pipeline.corefs.bert_corefs.CoreferenceDocument":[[2,2,1,"","__eq__"],[2,4,1,"","__hash__"],[2,2,1,"","__init__"],[2,2,1,"","__repr__"],[2,2,1,"","document_labels"],[2,2,1,"","from_labels"],[2,2,1,"","from_wpieced_to_tokenized"],[2,2,1,"","prepared_document"]],"renard.pipeline.corefs.bert_corefs.DataCollatorForSpanClassification":[[2,2,1,"","__eq__"],[2,4,1,"","__hash__"],[2,2,1,"","__init__"],[2,2,1,"","__repr__"]],"renard.pipeline.corefs.corefs":[[2,1,1,"","BertCoreferenceResolver"]],"renard.pipeline.corefs.corefs.BertCoreferenceResolver":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","production"]],"renard.pipeline.graph_extraction":[[2,1,1,"","CoOccurencesGraphExtractor"]],"renard.pipeline.graph_extraction.CoOccurencesGraphExtractor":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","_extract_dynamic_graph"],[2,2,1,"","_extract_gephi_dynamic_graph"],[2,2,1,"","_extract_graph"],[2,2,1,"","needs"],[2,2,1,"","production"]],"renard.pipeline.ner":[[2,1,1,"","BertNamedEntityRecognizer"],[2,1,1,"","NEREntity"],[2,1,1,"","NLTKNamedEntityRecognizer"],[2,3,1,"","ner_entities"],[2,3,1,"","score_ner"]],"renard.pipeline.ner.BertNamedEntityRecognizer":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","production"],[2,2,1,"","wp_labels_to_token_labels"]],"renard.pipeline.ner.NEREntity":[[2,2,1,"","__eq__"],[2,4,1,"","__hash__"],[2,2,1,"","__init__"],[2,2,1,"","__repr__"],[2,4,1,"","tag"]],"renard.pipeline.ner.NLTKNamedEntityRecognizer":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","production"]],"renard.pipeline.preprocessing":[[2,1,1,"","CustomSubstitutionPreprocessor"]],"renard.pipeline.preprocessing.CustomSubstitutionPreprocessor":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","production"]],"renard.pipeline.stanford_corenlp":[[2,1,1,"","StanfordCoreNLPPipeline"],[2,3,1,"","corenlp_annotations_bio_tags"]],"renard.pipeline.stanford_corenlp.StanfordCoreNLPPipeline":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","production"]],"renard.pipeline.tokenization":[[2,1,1,"","BertTokenizer"],[2,1,1,"","NLTKWordTokenizer"]],"renard.pipeline.tokenization.BertTokenizer":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","production"],[2,2,1,"","wp_tokens_to_tokens"]],"renard.pipeline.tokenization.NLTKWordTokenizer":[[2,2,1,"","__call__"],[2,2,1,"","__init__"],[2,2,1,"","needs"],[2,2,1,"","production"]],"renard.resources.hypocorisms":[[3,1,1,"","HypocorismGazetteer"]],"renard.resources.hypocorisms.HypocorismGazetteer":[[3,2,1,"","__init__"],[3,2,1,"","_add_hypocorism_"],[3,2,1,"","are_related"],[3,2,1,"","get_nicknames"],[3,2,1,"","get_possible_names"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"0":[2,3],"1":2,"10":2,"100":2,"128":2,"2":[2,3],"2003":2,"2012":2,"2015":2,"2017":2,"2019":2,"25":2,"3":2,"4":2,"5":2,"639":2,"8g":2,"9999999":2,"abstract":2,"case":2,"class":[2,3],"default":2,"do":2,"export":2,"final":2,"float":2,"function":2,"import":2,"int":2,"return":[2,3],"short":2,"static":2,"throw":2,"true":2,"try":2,A:2,If:[1,2],In:2,Is:2,It:2,The:1,To:2,__call__:2,__delattr__:2,__eq__:2,__hash__:2,__init__:[2,3],__repr__:2,__setattr__:2,_add_hypocorism_:3,_assign_coreference_ment:2,_extract_dynamic_graph:2,_extract_gephi_dynamic_graph:2,_extract_graph:2,_must_:2,accept:2,ad:2,adapt:2,add:[2,3],addit:2,additional_hypocor:2,al:2,algorithm:2,align:2,all:[2,3],alloc:2,allow:2,alreadi:2,also:2,an:[2,3],ani:2,annot:2,annotate_coref:2,anteced:2,antecedents_nb:2,apach:3,apparit:2,appear:2,appli:2,applyfinegrain:2,ar:[2,3],are_rel:3,arg:2,arrai:2,assign:2,associ:[2,3],attent:2,attention_mask:2,attribut:2,auto:2,b:2,base:2,basic:2,basictokenizerstep:2,batch:2,batch_idx:2,batch_siz:2,batchencod:2,be_quiet:2,becaus:2,befor:2,being:2,bert_batch_encod:2,bert_coref:2,bert_paramet:2,bertcoreferenceresolutionoutput:2,bertcoreferenceresolv:2,bertforcoreferenceresolut:2,bertnamedentityrecogn:2,berttoken:2,best:2,between:2,bio:2,bio_tag:2,block:2,block_siz:2,bool:[2,3],both:[2,3],c:2,ca:2,call:2,callabl:2,can:[1,2,3],candid:2,carltonnorthern:3,certain:2,chain:2,chapter:2,chapter_token:2,charact:0,character_graph:2,characters_extract:2,characters_graph:2,check:[2,3],check_valid:2,client_properti:2,closest:2,closest_antecedents_index:2,co:2,co_occurences_dist:2,code:2,collat:2,column:2,com:3,come:3,comput:2,config:2,conll:2,consid:2,construct:2,contain:2,convert:2,cooccurencesgraphextractor:2,core:0,coref:2,coref_chain:2,corefer:0,coreferencedataset:2,coreferencedocu:2,corefs_algorithm:2,corefs_split_idx:2,corenlp:[0,1],corenlp_annotations_bio_tag:2,corenlp_custom_properti:2,correct:2,correspond:3,cpu:2,cuda:2,cumul:2,current:2,custom:2,customsubstitutionpreprocessor:2,cut:2,data:[2,3],datacollatorforspanclassif:2,dataset:2,declar:2,def:2,defaul:2,delattr:2,depend:1,deriv:2,descript:2,detail:2,detect:2,determinist:2,devic:2,dict:2,dictionari:2,diminut:3,directli:2,directori:2,discard:2,distanc:2,document:2,document_label:2,doe:2,doesn:2,download:2,draw:2,draw_graph:2,draw_graph_to_fil:2,draw_graphs_to_dir:2,dslim:2,dure:2,dweight:2,dynam:2,dynamic_overlap:2,dynamic_window:2,e2ecoref:2,e:[1,2],each:2,easili:2,either:2,element:2,en:2,encod:2,end:2,end_idx:2,eng:2,entir:2,entiti:0,environ:1,equal:3,error:2,escap:2,et:2,ever:1,exampl:2,except:2,execut:2,expect:2,export_graph_to_gexf:2,extra:[1,2],extract:0,extractor:2,f1:2,f:2,fals:2,fig:2,figur:2,file:2,first:2,flexibl:2,follow:2,form:2,format:2,forward:2,four:2,from:[2,3],from_conll2012_fil:2,from_label:2,from_pretrain:2,from_wpieced_to_token:2,full:2,futur:2,g:2,gazeet:3,gazett:3,gender:2,gephi:2,get:[1,2],get_nicknam:3,get_possible_nam:3,gexf:2,github:[2,3],given:[2,3],graph:0,graph_extract:2,graph_start_idx:2,graph_with_nam:2,graphrulescharactersextractor:2,h:2,ha:2,harmon:2,hash:2,hatch:2,have:2,head_mask:2,help:2,here:2,hidden_s:2,hidden_st:2,high:2,how:2,html:2,http:[2,3],hugginfac:2,huggingfac:2,huggingface_model_id:2,hypocor:[0,2],hypocorismgazett:3,i:2,id:2,implement:2,index:[0,2],individu:2,infer:2,infer_name_gend:2,inform:2,init:2,initi:2,input:2,input_id:2,inputs_emb:2,inspir:2,instal:[0,2],instanti:2,instead:2,intend:2,interact:2,intern:2,intract:2,io:2,iro:2,iso:2,issu:2,iter:2,its:2,j:2,joshi:2,k:2,keep:2,kei:2,kept:2,kernel:2,kwarg:2,label:2,label_pad_token_id:2,languag:2,last:2,layer:2,layout:2,least:2,lee:2,letter:2,librari:2,licens:3,lifetim:2,like:2,limit:2,link:2,list:[2,3],liter:2,load:2,load_wikicoref_dataset:2,locat:2,logit:2,longest:2,longtensor:2,lookup:3,loos:2,loss:2,m:2,mai:2,manag:1,manual:2,match:2,matplotlib:2,max:2,max_char_length:2,max_length:2,max_span_len:2,max_span_s:2,maximum:2,mean:2,memori:2,mention:2,mention_compatibility_scor:2,mention_scor:2,mentions_per_token:2,merg:2,merged_dataset:2,messag:2,method:2,might:2,min_appear:2,minimum:2,misc:2,modul:[0,2],more:2,most:2,ms:2,must:2,my_doc:2,my_script:1,my_tokenization_funct:2,n:2,naivecharactersextractor:2,name1:3,name2:3,name:[0,3],name_styl:2,need:2,neeed:2,ner:2,ner_ent:2,nerent:2,network:2,networkx:2,neural:2,newli:2,nice:2,nicknam:[2,3],nlp:2,nltk:2,nltknamedentityrecogn:2,nltkwordtoken:2,nn:2,nnp:2,node:2,non:2,none:2,normal:2,notat:2,note:2,novel:2,number:2,nx:2,occur:2,one:[2,3],onli:2,open:2,option:2,optional_ne:2,order:2,organ:2,origin:2,os:2,other:[2,3],otherwis:2,out:2,output:2,output_attent:2,output_hidden_st:2,overlap:2,overriden:2,p:2,pad:2,page:0,paper:2,paramet:[2,3],particular:2,pass:2,path:2,per:2,percentag:2,person:2,piec:2,pipelin:[0,1],pipelinest:2,pipelinestep:2,po:2,poetri:[1,2],posit:2,position_id:2,possibl:[2,3],preced:2,precis:2,pred_bio_tag:2,pred_scor:2,predict:2,prefix:2,prepar:2,prepared_docu:2,preprocess:0,preprocessor:2,pretrainedtokenizerfast:2,previou:2,previous:2,process:2,produc:2,product:2,progress:2,progress_report:2,project:1,pronoun:2,propag:2,properti:2,prune:2,pruned_mentions_index:2,pt:2,py:1,pyplot:2,python:1,q:2,quick:0,quiet:2,rali:2,ram:2,read:2,reason:2,recal:2,recogn:2,recognit:0,ref_bio_tag:2,refer:2,regex:2,regular:2,relabel:2,renard:[2,3],report:2,repositori:2,repr:2,repres:2,requir:2,resolut:0,resolv:2,resolve_inconsist:2,resort:2,resourc:0,result:2,retoken:2,return_dict:2,return_tensor:2,root_path:2,rule:2,run:[1,2],runtim:2,s:2,same:2,satisfi:2,save:2,score:2,score_n:2,script:1,scriptmodul:2,search:0,second:2,section:2,see:2,seem:2,self:2,seq_siz:2,seqev:2,sequenc:2,sequenti:2,seri:2,server:2,server_kwarg:2,server_timeout:2,set:[2,3],setattr:2,sever:[2,3],shall:2,shape:2,share:2,shell:1,shortest:2,should:2,simpl:2,sinc:2,singl:2,size:2,slider:2,so:2,some:2,sort:2,space:2,span:2,span_bound:2,spans_nb:2,special:2,specifi:2,split:2,stable_layout:2,stanford:[0,1],stanford_corenlp:2,stanfordcorenlppipelin:2,stanfordnlp:2,stanza:[1,2],start:[0,2],start_idx:2,statist:2,store:2,str:[2,3],string:2,style:2,substit:2,substition_rul:2,substitut:2,support:2,t:2,tag:2,task:2,task_paramet:2,tensor:2,termin:2,text:2,than:2,thank:2,thei:2,them:2,therefor:2,thi:[2,3],those:2,time:2,timeout:2,timestep:2,todo:2,togeth:2,token:0,token_type_id:2,tokenization_utils_bas:2,tokens_split_idx:2,top_antecedents_index:2,top_mentions_index:2,top_mentions_nb:2,torch:2,tqdm:2,transform:2,troubleshoot:2,tupl:2,two:2,txt:2,type:[2,3],umontr:2,under:[1,3],union:2,until:2,us:[1,2,3],usag:2,usual:2,vala:2,valid:2,valu:2,variabl:2,virtual:1,wan:2,want:1,warn:2,we:2,weight:2,weirdli:2,when:2,where:2,which:2,whose:2,wikicoref:2,window:2,without:2,word:2,word_token:2,wordpiec:2,work:2,worst:2,wp_bio_tag:2,wp_label:2,wp_labels_to_token_label:2,wp_token:2,wp_tokens_to_token:2,write_gexf:2,yet:2,you:[1,2],yourself:2},titles:["Welcome to Renard\u2019s documentation!","Installation","Pipeline","Resources"],titleterms:{"new":2,The:2,bert:2,charact:2,content:0,core:2,corefer:2,corenlp:2,creat:2,document:0,entiti:2,extract:2,graph:2,hypocor:3,indic:0,instal:1,model:2,name:2,object:2,pipelin:2,preprocess:2,quick:1,recognit:2,renard:0,resolut:2,resourc:3,s:0,stanford:2,start:1,state:2,step:2,tabl:0,token:2,welcom:0}})