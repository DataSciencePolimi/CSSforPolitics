
conn = new Mongo();
db = conn.getDB("TweetScraper");
var cur = db.tweet.find();

var obj;
while(cur.hasNext()){

    obj = cur.next();
    if(obj.tw_lang!='en')
      continue;
    if(obj.tw_retweet_count < 25)
      continue;
    if(obj.tw_full == 'undefined')
      continue;
    text = obj.tw_full
    var words = text.split(" ")
    if(words.length<5)
      continue
    trimmed_date = obj.datetime.substring(0,10)
    print(""+obj.ID+"~"+trimmed_date+"~"+obj.tw_full+"");

}
