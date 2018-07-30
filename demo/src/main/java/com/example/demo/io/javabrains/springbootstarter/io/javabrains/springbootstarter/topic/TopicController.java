package com.example.demo.io.javabrains.springbootstarter.io.javabrains.springbootstarter.topic;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;


@RestController //whichever class is a restcontroller returns in json form only
public class TopicController {

    @Autowired
    private TopicService topicService;

    public TopicController(TopicService topicService) {
        this.topicService = topicService;
    }

    @RequestMapping("/topics")
    public List<Topic> getAllTopics(){  //this is returned in json form
        return topicService.getAllTopics();
    }

    @RequestMapping("/topics/{id}")
    public Topic getTopic(@PathVariable String id)
    {
        return topicService.getTopic(id);
    }
}
