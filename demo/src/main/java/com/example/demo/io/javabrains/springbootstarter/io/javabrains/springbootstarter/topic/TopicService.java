package com.example.demo.io.javabrains.springbootstarter.io.javabrains.springbootstarter.topic;

import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

@Service
public class TopicService {

    private List<Topic> topics = Arrays.asList(
            new Topic("Spring", "Spring Framework","spring framework description"),
            new Topic(" Java", "Cre Java","Core Java description"),
            new Topic("javascript", "Javascript","Javascript description")
    );

    public List<Topic> getAllTopics()
    {
        return topics;
    }

    public Topic getTopic(String id){

        return topics.stream().filter(t->id.equals(t.getId())).findFirst().get();
    }
}
